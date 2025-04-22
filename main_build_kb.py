# main_build_kb.py
# Entry point script to build the vector knowledge base.
# Imports and calls the main function from src.build_knowledge_base.

import time
from src.build_knowledge_base import build_and_save_vector_store

if __name__ == "__main__":
    print("Starting Knowledge Base build process...")
    start_time = time.time()

    build_and_save_vector_store()

    end_time = time.time()
    print(f"Knowledge Base build process finished in {end_time - start_time:.2f} seconds.")