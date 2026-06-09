import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset
import json
import os
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk
import logging

# Download required NLTK data (only needs to be done once)
# Uncomment these lines if running for the first time
# nltk.download('punkt')
# nltk.download('punkt_tab') # Might be needed depending on nltk version

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_PATH = "data/sickle_cell_clinical_notes.csv"
MODEL_NAME = "t5-small" # Use T5-small
OUTPUT_DIR = "../model/fine_tuned_t5_small_sickle_cell_gpu" # Updated output dir
MAX_LENGTH_INPUT = 256 # Max length for the input (prompt)
MAX_LENGTH_OUTPUT = 256 # Max length for the target (clinical note)
MAX_LENGTH_TOTAL = MAX_LENGTH_INPUT + MAX_LENGTH_OUTPUT # Total max length if needed elsewhere
TEST_SIZE = 0.10
VAL_SIZE = 0.10
EPOCHS = 2 # Reduced epochs for speed target
BATCH_SIZE_PER_DEVICE = 8 # Increased batch size for T5-small, adjust if OOM
GRADIENT_ACCUMULATION_STEPS = 1 # Keep simple initially, adjust if needed
LEARNING_RATE = 3e-5 # Slightly different LR often works well for T5
SAVE_STRATEGY = "epoch"
EVALUATION_STRATEGY = "epoch"
LOGGING_STEPS = 20
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
REPORT_TO = None
FP16 = True # Crucial for speed on RTX 3060
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
    # This function is less used now as input/target are handled separately in the dataset
    # But kept for potential use in evaluation or inference prompt creation
    inputs = df.apply(lambda row: f"Patient Summary: Pain Intensity (1-10): {row['pain_intensity']}, Hemoglobin (g/dL): {row['hemoglobin']}, Oxygen Saturation (%): {row['oxygen_saturation']}, Pain Type: {row['pain_type']}, Facility Type: {row['facility_type']}, Location: {row['location']}, Admitted: {row['admitted']}. Generate Clinical Note:", axis=1)
    targets = df['clinical_note']
    return inputs.tolist(), targets.tolist()


def calculate_perplexity(trainer, eval_dataset):
    """Calculates perplexity for Seq2Seq models. This is less standard than for Causal LM."""
    # Perplexity calculation is more complex for Seq2Seq models as the loss
    # is typically calculated only on the decoder output (labels).
    # A common proxy is to calculate the exponential of the average label loss.
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
    """Calculates BLEU and ROUGE scores on a sample using the T5 model for generation."""
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
    logger.info("Starting T5 training and evaluation process...")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} records.")

    # 2. Split Data
    logger.info(f"Splitting data into Train/Validation/Test...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42)

    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")

    # 3. Initialize Tokenizer and Model
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # T5 tokenizer might not have a pad token by default, add it if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        # Need to resize model's token embedding layer to account for the new pad token
        # This is usually done *after* loading the model but *before* training
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.resize_token_embeddings(len(tokenizer)) # Resize embeddings after adding pad token
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


    # 4. Create Datasets
    logger.info("Creating datasets...")
    train_dataset = SickleCellDataset(train_df, tokenizer, MAX_LENGTH_INPUT, MAX_LENGTH_OUTPUT)
    val_dataset = SickleCellDataset(val_df, tokenizer, MAX_LENGTH_INPUT, MAX_LENGTH_OUTPUT)
    test_dataset = SickleCellDataset(test_df, tokenizer, MAX_LENGTH_INPUT, MAX_LENGTH_OUTPUT)

    # 5. Data Collator for Seq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    # 6. Training Arguments
    total_train_samples = len(train_dataset)
    steps_per_epoch = total_train_samples // (BATCH_SIZE_PER_DEVICE * GRADIENT_ACCUMULATION_STEPS)
    if total_train_samples % (BATCH_SIZE_PER_DEVICE * GRADIENT_ACCUMULATION_STEPS) != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * EPOCHS
    logger.info(f"Calculated: {steps_per_epoch} steps per epoch, Total steps for {EPOCHS} epochs: {total_steps}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        eval_strategy=EVALUATION_STRATEGY,
        eval_steps=None,
        save_steps=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # Minimize eval loss
        greater_is_better=False,
        report_to=REPORT_TO,
        fp16=FP16,
        dataloader_pin_memory=True,
        # dataloader_num_workers=2, # Test if beneficial
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator, # Use the Seq2Seq collator
    )

    # 8. Train the Model
    logger.info("Starting T5 training for full epochs...")
    trainer.train()
    logger.info("T5 Training completed for all specified epochs.")

    # 9. Save the Fine-Tuned Model
    logger.info(f"Saving final fine-tuned T5 model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Final T5 model saved successfully.")

    # 10. Evaluate the Model (on Validation and Test sets *after* full training)
    logger.info("Starting FINAL evaluation on Validation Set...")
    val_perplexity = calculate_perplexity(trainer, val_dataset)
    logger.info("Calculating metrics on a sample of Validation Set...")
    try:
        val_metrics = calculate_bleu_rouge_sample(trainer, val_dataset, tokenizer, sample_size=100)
    except Exception as e:
        logger.error(f"Error calculating BLEU/ROUGE on Validation set: {e}")
        val_metrics = {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    logger.info("Starting FINAL evaluation on Test Set...")
    test_perplexity = calculate_perplexity(trainer, test_dataset)
    logger.info("Calculating metrics on a sample of Test Set...")
    try:
        test_metrics = calculate_bleu_rouge_sample(trainer, test_dataset, tokenizer, sample_size=100)
    except Exception as e:
        logger.error(f"Error calculating BLEU/ROUGE on Test set: {e}")
        test_metrics = {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    # 11. Prepare Results
    results = {
        "model_name": MODEL_NAME,
        "training_args": {
            "epochs": EPOCHS,
            "batch_size_per_device": BATCH_SIZE_PER_DEVICE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE_PER_DEVICE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "max_length_input": MAX_LENGTH_INPUT,
            "max_length_output": MAX_LENGTH_OUTPUT,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "fp16": FP16,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps
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

    # 12. Save Results
    output_file = "../evaluation_results_t5_gpu.json" # Updated output file name
    logger.info(f"Saving final evaluation results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("Final T5 results saved.")

    print("\n--- Final T5 Evaluation Results ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
