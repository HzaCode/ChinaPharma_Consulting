# main_evaluate_llm.py
# Entry point script to evaluate the performance of the fine-tuned LLM directly.
# Imports and calls the evaluation function from src.evaluators.

import time
from src.evaluators import evaluate_llm_performance

if __name__ == "__main__":
    print("Starting Fine-tuned LLM direct evaluation...")
    start_time = time.time()

    results = evaluate_llm_performance()

    end_time = time.time()
    print(f"\nLLM evaluation finished in {end_time - start_time:.2f} seconds.")

    if results:
        print("\n--- Evaluation Summary ---")
        # Example: Print ROUGE-L score if available
        rouge_l_score = results.get('rougeL', None)
        if rouge_l_score is not None:
            print(f"ROUGE-L Score: {rouge_l_score*100:.2f}")
        else:
            print("Could not retrieve main evaluation score.")
    else:
        print("Evaluation did not produce results.")