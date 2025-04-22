# main_evaluate_rag.py
# Entry point script to evaluate the performance of the end-to-end RAG pipeline.
# Imports and calls the evaluation function from src.evaluators.

import time
from src.evaluators import evaluate_rag_performance

if __name__ == "__main__":
    print("Starting RAG Pipeline end-to-end evaluation...")
    start_time = time.time()

    results = evaluate_rag_performance()

    end_time = time.time()
    print(f"\nRAG evaluation finished in {end_time - start_time:.2f} seconds.")

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