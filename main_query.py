# main_query.py
# Entry point script to run the interactive RAG query interface.
# Loads the RAG pipeline and allows users to ask questions.

import time
from src.rag_pipeline import RAGPipeline

def run_interactive_query():
    """Starts the interactive command-line query session."""
    print("Initializing RAG system for querying...")
    try:
        rag_system = RAGPipeline()
        print("\n--- China Pharmacopoeia Consulting System ---")
        print("Enter your query below. Type 'exit' or 'quit' to end.")

        while True:
            query = input("\nYour Query: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting system. Goodbye!")
                break
            if not query.strip():
                continue

            start_time = time.time()
            answer = rag_system.answer_query(query)
            end_time = time.time()

            print("\n--- Answer ---")
            print(answer)
            print(f"(Answer generated in {end_time - start_time:.2f} seconds)")

    except RuntimeError as e:
         print(f"\nFATAL ERROR during initialization: {e}")
         print("Cannot start interactive query session.")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_interactive_query()