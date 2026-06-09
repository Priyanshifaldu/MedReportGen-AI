import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import json
import os
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk
import logging

# Download required NLTK data (only needs to be done once per environment)
# Run these commands in your Python environment if you haven't already:
# import nltk
nltk.download('punkt')
nltk.download('punkt_tab') 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Point this to the directory where your model was saved
MODEL_PATH = r"D:\VScodefiles\MedReportGen-AI\backend\model\fine_tuned_t5_small_sickle_cell_gpu" # Update if using a different path
DATA_PATH = r"D:\VScodefiles\MedReportGen-AI\backend\data\sickle_cell_clinical_notes.csv"
MAX_LENGTH_INPUT = 256 # Must match the lengths used during training
MAX_LENGTH_OUTPUT = 256
TEST_SIZE = 0.10
VAL_SIZE = 0.10
# Evaluation settings
EVAL_BATCH_SIZE = 8 # Can often use a larger batch size for evaluation
SAMPLE_SIZE_FOR_METRICS = 100 # Number of samples to calculate BLEU/ROUGE on (to save time)
# --- End Configuration ---

class SickleCellDataset(Dataset):
    def __init__(self, df, tokenizer, max_length_input, max_length_output):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        # Prepare the full text for tokenization
        self.inputs = df.apply(lambda row: f"Patient Summary: Pain Intensity (1-10): {row['pain_intensity']}, Hemoglobin (g/dL): {row['hemoglobin']}, Oxygen Saturation (%): {row['oxygen_saturation']}, Pain Type: {row['pain_type']}, Facility Type: {row['facility_type']}, Location: {row['location']}, Admitted: {row['admitted']}. Generate Clinical Note:", axis=1).tolist()
        self.targets = df['clinical_note'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = str(self.inputs[idx])
        target_text = str(self.targets[idx])

        # Tokenize input
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length_input,
            return_tensors='pt'
        )

        # Tokenize target
        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length_output,
            return_tensors='pt'
        )

        # Labels are the target token IDs, but the model calculates loss internally
        # We need to replace padding token id's of the labels by -100 so that the loss function ignores padding tokens
        labels = target_encodings['input_ids'].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encodings['input_ids'].flatten(),
            'attention_mask': input_encodings['attention_mask'].flatten(),
            'labels': labels # Use the processed labels tensor
        }

def prepare_prompts(df):
    """Formats structured data into an input prompt string for T5."""
    inputs = df.apply(lambda row: f"Patient Summary: Pain Intensity (1-10): {row['pain_intensity']}, Hemoglobin (g/dL): {row['hemoglobin']}, Oxygen Saturation (%): {row['oxygen_saturation']}, Pain Type: {row['pain_type']}, Facility Type: {row['facility_type']}, Location: {row['location']}, Admitted: {row['admitted']}. Generate Clinical Note:", axis=1)
    targets = df['clinical_note']
    return inputs.tolist(), targets.tolist()

def calculate_perplexity(trainer, eval_dataset):
    """Calculates perplexity for Seq2Seq models. This is a proxy."""
    trainer.model.eval()
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = trainer.model(**batch)
            loss = outputs.loss # This is the average loss per *token* in the labels for T5
            # Get the number of non-pad tokens in the labels for this batch
            labels = batch['labels']
            non_pad_mask = labels != -100
            num_tokens_in_batch = non_pad_mask.sum().item()

            total_loss += loss.item() * num_tokens_in_batch # Multiply average loss by token count
            total_tokens += num_tokens_in_batch

    if total_tokens == 0:
        logger.warning("Total tokens in eval set is 0. Cannot calculate perplexity.")
        return float('inf')

    avg_loss_per_token = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss_per_token)).item()
    return perplexity

def calculate_bleu_rouge_sample(trainer, eval_dataset, tokenizer, sample_size=50):
    """Calculates BLEU and ROUGE scores on a sample using the loaded model for generation."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    generated_texts = []
    target_texts = []

    # Iterate through a sample of the eval dataset for generation
    sample_indices = list(range(min(sample_size, len(eval_dataset))))
    logger.info(f"Calculating BLEU/ROUGE on {len(sample_indices)} samples...")

    for i in sample_indices:
        # Get the input_ids and attention_mask for the sample
        item = eval_dataset[i]
        input_ids = item['input_ids'].unsqueeze(0).to(trainer.model.device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(trainer.model.device)
        # Get the target text for comparison
        target_text = eval_dataset.targets[i] # Access target from the dataset object

        # Generate continuation using the model
        with torch.no_grad():
            outputs = trainer.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_LENGTH_OUTPUT, # Limit generated tokens to target length
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_time=10.0 # Timeout
            )

        # Decode the generated output
        generated_text_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The generated text *should* be just the clinical note part
        generated_note = generated_text_full.strip()

        generated_texts.append(generated_note)
        target_texts.append(target_text)

    # Calculate BLEU and ROUGE scores
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for gen, tgt in zip(generated_texts, target_texts):
        if gen.strip() == "" or tgt.strip() == "":
             logger.warning(f"Empty generated or target text found. Skipping BLEU/ROUGE for this pair.")
             continue # Skip empty texts to avoid errors
        # BLEU
        gen_tokens = nltk.word_tokenize(gen.lower())
        tgt_tokens = [nltk.word_tokenize(tgt.lower())] # BLEU expects list of references
        try:
            bleu_score = sentence_bleu(tgt_tokens, gen_tokens)
            bleu_scores.append(bleu_score)
        except Exception as e:
            logger.warning(f"Error calculating BLEU for pair: {e}. Skipping.")
            continue # Skip if BLEU calculation fails

        # ROUGE
        try:
            rouge_score = scorer.score(tgt, gen)
            rouge_scores['rouge1'].append(rouge_score['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(rouge_score['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(rouge_score['rougeL'].fmeasure)
        except Exception as e:
            logger.warning(f"Error calculating ROUGE for pair: {e}. Skipping.")
            continue # Skip if ROUGE calculation fails

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0
    avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0

    return {
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

def main():
    logger.info(f"Starting evaluation using model from: {MODEL_PATH}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading tokenizer and model from: {MODEL_PATH}")
    try:
        # Use MODEL_PATH instead of MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # Check if pad_token exists, add if necessary (should match training)
        if tokenizer.pad_token is None:
            logger.warning("Pad token not found in tokenizer, adding '<pad>'. Ensure this matches training setup.")
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

        # Use MODEL_PATH instead of MODEL_NAME
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        # Resize embeddings if pad token was added
        if tokenizer.pad_token is None or tokenizer.pad_token_id >= model.get_input_embeddings().num_embeddings:
             model.resize_token_embeddings(len(tokenizer))

    except Exception as e:
        logger.error(f"Failed to load model/tokenizer from {MODEL_PATH}: {e}")
        return

    # --- Load Data ---
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} records.")

    # --- Split Data ---
    logger.info(f"Splitting data into Train/Validation/Test...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42)

    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")

    # --- Create Datasets ---
    logger.info("Creating validation and test datasets...")
    val_dataset = SickleCellDataset(val_df, tokenizer, MAX_LENGTH_INPUT, MAX_LENGTH_OUTPUT)
    test_dataset = SickleCellDataset(test_df, tokenizer, MAX_LENGTH_INPUT, MAX_LENGTH_OUTPUT)

    # --- Setup Dummy Trainer for Evaluation ---
    # We create a minimal trainer just to use its evaluation helpers
    # We don't provide a train dataset or run training
    training_args = TrainingArguments(
        output_dir="./temp_eval_output", # Dummy output dir
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to=None, # Disable reporting
        # fp16=torch.cuda.is_available(), # Consider using fp16 for eval speed if training used it
        # Use the same fp16 setting as training if desired
        fp16=True # Assuming training used fp16, keep it for consistency/speed
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    dummy_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        # No train_dataset or eval_dataset needed here, we pass them explicitly later
    )

    # --- Evaluate Validation Set ---
    logger.info("Starting evaluation on Validation Set...")
    val_perplexity = calculate_perplexity(dummy_trainer, val_dataset)
    logger.info("Calculating metrics on a sample of Validation Set...")
    try:
        val_metrics = calculate_bleu_rouge_sample(dummy_trainer, val_dataset, tokenizer, sample_size=SAMPLE_SIZE_FOR_METRICS)
    except Exception as e:
        logger.error(f"Error calculating BLEU/ROUGE on Validation set: {e}")
        val_metrics = {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    # --- Evaluate Test Set ---
    logger.info("Starting evaluation on Test Set...")
    test_perplexity = calculate_perplexity(dummy_trainer, test_dataset)
    logger.info("Calculating metrics on a sample of Test Set...")
    try:
        test_metrics = calculate_bleu_rouge_sample(dummy_trainer, test_dataset, tokenizer, sample_size=SAMPLE_SIZE_FOR_METRICS)
    except Exception as e:
        logger.error(f"Error calculating BLEU/ROUGE on Test set: {e}")
        test_metrics = {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    # --- Prepare and Save Results ---
    results = {
        "model_path": MODEL_PATH,
        "evaluation_args": {
            "max_length_input": MAX_LENGTH_INPUT,
            "max_length_output": MAX_LENGTH_OUTPUT,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "sample_size_for_metrics": SAMPLE_SIZE_FOR_METRICS,
            "fp16_used": True # Assuming fp16 was used during training and is kept here
        },
        "validation": {
            "perplexity": val_perplexity, # Note: This is a proxy for T5
            "bleu": val_metrics['bleu'],
            "rouge1": val_metrics['rouge1'],
            "rouge2": val_metrics['rouge2'],
            "rougeL": val_metrics['rougeL']
        },
        "test": {
            "perplexity": test_perplexity, # Note: This is a proxy for T5
            "bleu": test_metrics['bleu'],
            "rouge1": test_metrics['rouge1'],
            "rouge2": test_metrics['rouge2'],
            "rougeL": test_metrics['rougeL']
        }
    }

    output_file = f"../final_evaluation_results_{os.path.basename(MODEL_PATH)}.json" # Create a unique name based on model path
    logger.info(f"Saving final evaluation results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("Final evaluation results saved.")

    print("\n--- Final Evaluation Results ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    # The MODEL_PATH variable defined above is used within the script
    main()