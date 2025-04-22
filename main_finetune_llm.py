# main_finetune_llm.py
# Entry point script to start the LLM fine-tuning process.
# Imports and calls the main function from src.llm_finetuner.

import time
from src.llm_finetuner import fine_tune_llm

if __name__ == "__main__":
    print("Starting LLM Fine-tuning process...")
    start_time = time.time()

    fine_tune_llm()

    end_time = time.time()
    print(f"LLM Fine-tuning process finished in {end_time - start_time:.2f} seconds.")