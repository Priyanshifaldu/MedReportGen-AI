# backend/app/model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os

logger = logging.getLogger(__name__)

class ClinicalNoteGenerator:
    def __init__(self, model_path: str):
        """
        Initializes the generator by loading the pre-trained/fine-tuned model and tokenizer.
        Args:
            model_path (str): Path to the directory containing the saved model files (config.json, pytorch_model.bin, etc.).
        """
        logger.info(f"Loading model from {model_path}")
        try:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load the model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            # Ensure pad token exists (matching training setup)
            if self.tokenizer.pad_token is None:
                logger.warning("Pad token not found in tokenizer, adding '<pad>'. Ensure this matches training setup.")
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                # Resize model embeddings if pad token was added
                self.model.resize_token_embeddings(len(self.tokenizer))

            # Set model to evaluation mode (important for inference)
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer from {model_path}: {e}")
            raise # Re-raise the exception to halt startup if model fails to load

    def generate_note(self, summary: dict) -> str:
        """
        Generates a clinical note based on the patient summary.
        Args:
            summary (dict): A dictionary containing patient information (pain_intensity, hemoglobin, etc.).
        Returns:
            str: The generated clinical note.
        """
        # Construct the prompt exactly as done during training/evaluation for T5
        # Using the format defined in the SickleCellDataset.prepare_prompts equivalent
        prompt = f"Patient Summary: Pain Intensity (1-10): {summary['pain_intensity']}, Hemoglobin (g/dL): {summary['hemoglobin']}, Oxygen Saturation (%): {summary['oxygen_saturation']}, Pain Type: {summary['pain_type']}, Facility Type: {summary['facility_type']}, Location: {summary['location']}, Admitted: {summary['admitted']}. Generate Clinical Note:"

        logger.info(f"Generating note with prompt: {prompt[:100]}...") # Log first 100 chars of prompt

        try:
            # Tokenize the input prompt
            input_encodings = self.tokenizer(
                prompt,
                truncation=True,
                padding='max_length',
                max_length=256, # Use the same length as during training/evaluation
                return_tensors='pt'
            )

            input_ids = input_encodings['input_ids']
            attention_mask = input_encodings['attention_mask']

            # Generate the clinical note using the model
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256, # Use the same length as during training/evaluation
                    temperature=0.7,   # Add some randomness
                    do_sample=True,    # Enable sampling
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # max_time=10.0 # Optional timeout
                )

            # Decode the generated output IDs back to text
            generated_text_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated note part (everything after the prompt)
            # This assumes the prompt format is consistent
            note_start_marker = "Generate Clinical Note:"
            note_start_idx = generated_text_full.rfind(note_start_marker)
            if note_start_idx != -1:
                generated_note = generated_text_full[note_start_idx + len(note_start_marker):].strip()
            else:
                # Fallback: return the whole generated text if marker not found
                logger.warning(f"Prompt marker '{note_start_marker}' not found in output. Returning full output.")
                generated_note = generated_text_full.strip()

            logger.info(f"Generated note: {generated_note[:100]}...") # Log first 100 chars
            return generated_note

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating note: {str(e)}"


# --- Global Generator Instance ---
# This is the instance that main.py will import
# The path is relative to where the app runs inside the Docker container (/app/)
# Adjust 'fine_tuned_t5_small_sickle_cell_gpu' if your saved model folder has a different name
MODEL_PATH = "model/fine_tuned_t5_small_sickle_cell_gpu"
generator = ClinicalNoteGenerator(model_path=MODEL_PATH)
# --- End Global Generator Instance ---