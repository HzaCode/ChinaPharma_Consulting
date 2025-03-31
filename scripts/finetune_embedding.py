import logging
import argparse
import os
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import math

# --- Configuration ---
DEFAULT_BASE_MODEL = 'BAAI/bge-small-zh-v1.5'
DEFAULT_OUTPUT_DIR = "./output" 
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 16 adjust based on GPU memory
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1 # 10% warmup steps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(file_path, delimiter='\t', quotechar='"', skip_header=False):
    """Loads training data (sentence pairs) from a TSV/CSV file."""
    examples = []
    try:
        with open(file_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            if skip_header:
                next(reader) # Skip header row
            for i, row in enumerate(reader):
                if len(row) >= 2:
                    examples.append(InputExample(texts=[row[0], row[1]]))
                else:
                    logger.warning(f"Skipping invalid row {i+1} in {file_path}: Expected 2 columns, got {len(row)}")
        logger.info(f"Loaded {len(examples)} training examples from {file_path}")
        return examples
    except FileNotFoundError:
        logger.error(f"Training data file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading training data from {file_path}: {e}", exc_info=True)
        return None

def load_evaluation_data(file_path, delimiter='\t', quotechar='"', skip_header=False):
    """Loads evaluation data (sentence1, sentence2, score) from a TSV/CSV file."""
    sentences1 = []
    sentences2 = []
    scores = []
    try:
        with open(file_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            if skip_header:
                next(reader) # Skip header row
            for i, row in enumerate(reader):
                if len(row) >= 3:
                    try:
                        sentences1.append(row[0])
                        sentences2.append(row[1])
                        scores.append(float(row[2]))
                    except ValueError:
                        logger.warning(f"Skipping invalid row {i+1} in {file_path}: Could not convert score '{row[2]}' to float.")
                    except Exception as inner_e:
                         logger.warning(f"Skipping invalid row {i+1} in {file_path} due to error: {inner_e}")
                else:
                    logger.warning(f"Skipping invalid row {i+1} in {file_path}: Expected 3 columns, got {len(row)}")

        if not sentences1:
            logger.warning(f"No valid evaluation examples loaded from {file_path}")
            return None, None, None

        logger.info(f"Loaded {len(scores)} evaluation examples from {file_path}")
        return sentences1, sentences2, scores
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found: {file_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error reading evaluation data from {file_path}: {e}", exc_info=True)
        return None, None, None

def main(args):
    logger.info("--- Starting Sentence Transformer Fine-tuning ---")

    # --- Setup Output Path ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_short = args.base_model.split('/')[-1] # Get last part of model name
    output_path = args.output_path or os.path.join(DEFAULT_OUTPUT_DIR, f"finetune-{model_name_short}-{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output path: {output_path}")

    # --- Load Base Model ---
    logger.info(f"Loading base model: {args.base_model}")
    try:
        model = SentenceTransformer(args.base_model)
    except Exception as e:
        logger.error(f"Failed to load base model '{args.base_model}'. Error: {e}", exc_info=True)
        return

    # --- Load Training Data ---
    logger.info(f"Loading training data from: {args.train_data_path}")
    train_samples = load_training_data(args.train_data_path, skip_header=args.skip_header)
    if not train_samples:
        logger.error("No training data loaded. Exiting.")
        return
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    num_training_steps = len(train_dataloader) * args.num_epochs

    # --- Define Loss ---
    # MultipleNegativesRankingLoss is good for pairs where text[0] and text[1] should be close
    # CosineSimilarityLoss is simpler if you have labels (e.g. 0/1 or continuous score) per pair
    logger.info(f"Using loss function: MultipleNegativesRankingLoss")
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # --- Setup Evaluator (Optional) ---
    evaluator = None
    evaluation_steps_calculated = 0
    if args.eval_data_path:
        logger.info(f"Loading evaluation data for intermediate evaluation from: {args.eval_data_path}")
        eval_s1, eval_s2, eval_scores = load_evaluation_data(args.eval_data_path, skip_header=args.skip_header)
        if eval_s1:
            evaluator_name = f"eval-{model_name_short}-{timestamp}"
            evaluator = EmbeddingSimilarityEvaluator(eval_s1, eval_s2, eval_scores, name=evaluator_name, write_csv=True)

            if args.evaluation_steps <= 0:
                # Evaluate roughly once per epoch by default if not specified
                steps_per_epoch = len(train_dataloader)
                evaluation_steps_calculated = max(1, steps_per_epoch)
                logger.info(f"Evaluation steps calculated dynamically (once per epoch): {evaluation_steps_calculated}")
            else:
                 evaluation_steps_calculated = args.evaluation_steps
                 logger.info(f"Evaluation steps set by argument: {evaluation_steps_calculated}")
        else:
             logger.warning("Could not load evaluation data, skipping intermediate evaluation.")
    else:
        logger.info("No evaluation data path provided, skipping intermediate evaluation during training.")

    # --- Calculate Warmup Steps ---
    warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps ({args.warmup_ratio*100}%): {warmup_steps}")

    # --- Configure Checkpointing ---
    checkpoint_save_steps = args.checkpoint_steps if args.checkpoint_steps > 0 else 0
    checkpoint_path = os.path.join(output_path, "checkpoints") if checkpoint_save_steps > 0 else None
    if checkpoint_save_steps > 0:
        logger.info(f"Saving checkpoints every {checkpoint_save_steps} steps to {checkpoint_path}")
    else:
        logger.info("Checkpoint saving disabled.")


    # --- Start Fine-tuning ---
    logger.info("Starting fine-tuning process...")
    try:
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=args.num_epochs,
                  evaluation_steps=evaluation_steps_calculated,
                  warmup_steps=warmup_steps,
                  output_path=output_path,
                  save_best_model=args.save_best_model and (evaluator is not None), # Only save best if evaluator exists
                  checkpoint_save_steps=checkpoint_save_steps,
                  checkpoint_path=checkpoint_path,
                  optimizer_params={'lr': args.learning_rate},
                  show_progress_bar=True)
        logger.info(f"Fine-tuning finished. Model saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

    # Final save (optional, fit usually saves the last state)
    try:
        model.save(output_path)
        logger.info(f"Ensured final model state is saved to {output_path}")
    except Exception as save_err:
        logger.error(f"Error explicitly saving final model state: {save_err}", exc_info=True)

    logger.info("--- Fine-tuning script completed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Sentence Transformer embedding model.")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the training data file (TSV/CSV: sentence1<TAB>sentence2).")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Name or path of the base Sentence Transformer model.")
    parser.add_argument("--output_path", type=str, default=None,
                        help=f"Directory where the fine-tuned model will be saved (defaults to '{DEFAULT_OUTPUT_DIR}/finetune-<model>-<timestamp>').")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="(Optional) Path to evaluation data file (TSV/CSV: sentence1<TAB>sentence2<TAB>score) for intermediate evaluation.")
    parser.add_argument("--skip_header", action='store_true',
                        help="Set this flag if your TSV/CSV data files contain a header row.")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="Ratio of total training steps used for warmup.")
    parser.add_argument("--evaluation_steps", type=int, default=0,
                        help="Evaluate performance every N training steps (0 to evaluate approx once per epoch, requires --eval_data_path).")
    parser.add_argument("--checkpoint_steps", type=int, default=0,
                        help="Save a checkpoint every N training steps (set > 0 to enable).")
    parser.add_argument("--save_best_model", action='store_true',
                        help="Save the best model found during intermediate evaluation (requires --eval_data_path).")


    args = parser.parse_args()
    main(args)
