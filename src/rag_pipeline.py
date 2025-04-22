# src/rag_pipeline.py
# Orchestrates the entire RAG process:
# Takes a user query, retrieves relevant context using the Retriever,
# and generates an answer using the LLMHandler with the retrieved context.

from . import config
from .retriever import VectorRetriever
from .llm_handler import LLMAnswerGenerator

class RAGPipeline:
    """Combines retrieval and generation to answer queries."""
    def __init__(self):
        """Initializes the retriever and LLM components."""
        print("--- Initializing RAG Pipeline ---")
        try:
            self.retriever = VectorRetriever()
            self.llm_generator = LLMAnswerGenerator()
            print("--- RAG Pipeline Initialized Successfully ---")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"FATAL: Failed to initialize RAG Pipeline components: {e}")
            # Decide how to handle this - maybe raise the exception
            # For now, set components to None to indicate failure
            self.retriever = None
            self.llm_generator = None
            raise RuntimeError("RAG Pipeline failed to initialize.") from e


    def answer_query(self, query: str) -> str:
        """
        Answers a query using the RAG process.
        1. Retrieve relevant chunks.
        2. Generate answer using LLM with retrieved chunks.
        """
        if not self.retriever or not self.llm_generator:
             return "错误：RAG 系统未正确初始化。"

        print(f"\n--- Answering Query: '{query}' ---")

        # 1. Retrieve Context
        retrieved_chunks = self.retriever.retrieve(query, top_k=config.TOP_K)

        if not retrieved_chunks:
            print("Warning: No relevant context found. Attempting to answer without context.")
            # Optionally, you could return a specific message here
            # or let the LLM try to answer based on its fine-tuned knowledge only.
            context_chunks_for_llm = [] # Pass empty context
        else:
            context_chunks_for_llm = retrieved_chunks

        # 2. Generate Answer
        final_answer = self.llm_generator.generate(query, context_chunks_for_llm)

        print("--- Query Answering Complete ---")
        return final_answer

if __name__ == '__main__':
    # Example usage
    try:
        rag_system = RAGPipeline()
        # sample_query = "什么是前言？"
        sample_query = "乙酰唑胺片的鉴别方法是什么？"
        answer = rag_system.answer_query(sample_query)
        print("\n================================")
        print(f"Query: {sample_query}")
        print(f"Final Answer:\n{answer}")
        print("================================")
    except RuntimeError as e:
        print(f"Could not run RAG pipeline example: {e}")