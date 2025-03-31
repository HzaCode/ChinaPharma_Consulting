import logging
import argparse
import os
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "./output/evaluation"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info("--- Starting Sentence Transformer Evaluation ---")

    # --- Setup Output Path ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_short = args.model_name_or_path.replace('/','_').split('_')[-1] # Try to get a readable name
    eval_output_path = args.output_path or os.path.join(DEFAULT_OUTPUT_DIR, f"{model_name_short}-{timestamp}")
    os.makedirs(eval_output_path, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {eval_output_path}")

    # --- Load Model ---
    logger.info(f"Loading model for evaluation: {args.model_name_or_path}")
    try:
        model = SentenceTransformer(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load model '{args.model_name_or_path}'. Error: {e}", exc_info=True)
        return

    # --- Load Evaluation Data ---
    logger.info(f"Loading evaluation data from: {args.eval_data_path}")
    eval_s1, eval_s2, eval_scores = load_evaluation_data(args.eval_data_path, skip_header=args.skip_header)
    if not eval_s1:
        logger.error("No evaluation data loaded. Cannot perform evaluation. Exiting.")
        return

    # --- Create Evaluator ---
    evaluator_name = args.evaluator_name or f"eval_{model_name_short}"
    evaluator = EmbeddingSimilarityEvaluator(
        eval_s1,
        eval_s2,
        eval_scores,
        name=evaluator_name,
        batch_size=args.batch_size,
        show_progress_bar=True,
        write_csv=True # Ensure the detailed CSV report is saved
    )

    # --- Run Evaluation ---
    logger.info(f"Running evaluation using '{evaluator_name}'...")
    final_eval_results = None
    try:
        # The evaluator writes the CSV to output_path and returns the main score (Spearman) or a dict
        final_eval_results = evaluator(model, output_path=eval_output_path)
        logger.info(f"Evaluation run complete. Evaluator returned: {final_eval_results}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
        return # Exit if evaluation fails

    # --- Print Summary ---
    print("\n" + "="*40)
    print(f"      EVALUATION SUMMARY: {args.model_name_or_path}")
    print("="*40)
    print(f"Model Evaluated: {args.model_name_or_path}")
    print(f"Evaluation Dataset: {args.eval_data_path}")
    print(f"Number of Evaluation Pairs: {len(eval_scores)}")
    print(f"Results Output Path: {eval_output_path}")
    print("-" * 40)

    final_spearman_score = None
    final_pearson_score = None

    # Determine keys based on common SBERT patterns
    spearman_key_cos = f"{evaluator.name}_spearman_cosine"
    pearson_key_cos = f"{evaluator.name}_pearson_cosine"
    spearman_key_euc = f"{evaluator.name}_spearman_euclidean" # Less common but possible
    pearson_key_euc = f"{evaluator.name}_pearson_euclidean"
    spearman_key_man = f"{evaluator.name}_spearman_manhattan"
    pearson_key_man = f"{evaluator.name}_pearson_manhattan"
    spearman_key_dot = f"{evaluator.name}_spearman_dot"
    pearson_key_dot = f"{evaluator.name}_pearson_dot"

    if isinstance(final_eval_results, dict):
        logger.info(f"Evaluation results dictionary: {final_eval_results}")
        # Prioritize cosine similarity results if available
        if spearman_key_cos in final_eval_results:
            final_spearman_score = final_eval_results[spearman_key_cos]
        elif spearman_key_euc in final_eval_results:
             final_spearman_score = final_eval_results[spearman_key_euc]
        elif spearman_key_man in final_eval_results:
             final_spearman_score = final_eval_results[spearman_key_man]
        elif spearman_key_dot in final_eval_results:
             final_spearman_score = final_eval_results[spearman_key_dot]
        else:
            logger.warning(f"Could not find a known Spearman score key in results dict.")

        if pearson_key_cos in final_eval_results:
            final_pearson_score = final_eval_results[pearson_key_cos]
        elif pearson_key_euc in final_eval_results:
             final_pearson_score = final_eval_results[pearson_key_euc]
        elif pearson_key_man in final_eval_results:
             final_pearson_score = final_eval_results[pearson_key_man]
        elif pearson_key_dot in final_eval_results:
             final_pearson_score = final_eval_results[pearson_key_dot]
        else:
             logger.warning(f"Could not find a known Pearson score key in results dict.")

    elif isinstance(final_eval_results, float): # Evaluator might just return the main score (Spearman by default)
         final_spearman_score = final_eval_results
         logger.info("Evaluator returned a single float score (assumed Spearman). Pearson score not available from direct return value.")
    else:
         logger.warning(f"Evaluator returned unexpected type: {type(final_eval_results)}. Cannot extract scores directly.")

    # Print extracted scores
    if final_spearman_score is not None:
        try:
            print(f"Spearman Correlation (Cosine/Primary): {final_spearman_score:.4f}")
        except (TypeError, ValueError):
             print(f"Spearman Correlation (Cosine/Primary): Error formatting score ({final_spearman_score})")
             logger.error(f"Spearman score formatting failed. Score type: {type(final_spearman_score)}, Value: {final_spearman_score}")
    else:
        print("Spearman Correlation (Cosine/Primary): Not found or evaluation failed.")

    if final_pearson_score is not None:
        try:
            print(f"Pearson Correlation (Cosine/Primary): {final_pearson_score:.4f}")
        except (TypeError, ValueError):
             print(f"Pearson Correlation (Cosine/Primary): Error formatting score ({final_pearson_score})")
             logger.error(f"Pearson score formatting failed. Score type: {type(final_pearson_score)}, Value: {final_pearson_score}")
    else:
        print("Pearson Correlation (Cosine/Primary): Not found or evaluation failed.")


    # Check for detailed results file explicitly
    eval_filename_csv = f"similarity_evaluation_{evaluator.name}_results.csv"
    eval_filepath = os.path.join(eval_output_path, eval_filename_csv)
    if os.path.exists(eval_filepath):
         print(f"\nDetailed results saved to: {eval_filepath}")
    else:
         print(f"\n(Note: Detailed CSV report '{eval_filepath}' may not have been generated if evaluation failed early)")


    print("="*40)
    logger.info("--- Evaluation script completed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained/fine-tuned Sentence Transformer model.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path or Hugging Face name of the model to evaluate.")
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="Path to the evaluation data file (TSV/CSV: sentence1<TAB>sentence2<TAB>score).")
    parser.add_argument("--output_path", type=str, default=None,
                        help=f"Directory where evaluation results (e.g., detailed CSV) will be saved (defaults to '{DEFAULT_OUTPUT_DIR}/<model>-<timestamp>').")
    parser.add_argument("--skip_header", action='store_true',
                        help="Set this flag if your TSV/CSV evaluation data file contains a header row.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--evaluator_name", type=str, default=None, help="Custom name for the evaluator (used in output filenames).")


    args = parser.parse_args()
    main(args)
