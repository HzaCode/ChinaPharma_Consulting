# src/retriever.py
# Handles loading the vector store and the embedding model
# to perform similarity searches and retrieve relevant document chunks.

# Import necessary libraries
from sentence_transformers import SentenceTransformer
# import faiss
# import chromadb
import numpy as np
import os
import pickle

# Import project modules
from . import config

class VectorRetriever:
    """
    Loads a pre-built vector store and embedding model to retrieve
    relevant text chunks based on a query.
    """
    def __init__(self):
        """Initializes the retriever by loading the model and vector store."""
        self.embedding_model = None
        self.index = None
        self.text_chunks = None
        self._load_resources()

    def _load_resources(self):
        """Loads the embedding model and vector store components."""
        print("--- Initializing Retriever ---")
        # 1. Load Embedding Model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        try:
            # Ensure trust_remote_code=True if needed
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE, trust_remote_code=True)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise RuntimeError("Failed to load embedding model for retriever.") from e

        # 2. Load Vector Store and Text Chunks (Example using FAISS index + pickle)
        index_file = os.path.join(config.VECTOR_STORE_PATH, "knowledge_base.index")
        chunks_file = os.path.join(config.VECTOR_STORE_PATH, "knowledge_base_chunks.pkl")
        # --- Placeholder for FAISS loading ---
        embeddings_file = os.path.join(config.VECTOR_STORE_PATH, "knowledge_base_embeddings.pkl")

        if os.path.exists(embeddings_file) and os.path.exists(chunks_file): # Check placeholder files
            print(f"Loading text chunks from: {chunks_file}")
            with open(chunks_file, 'rb') as f_chunks:
                self.text_chunks = pickle.load(f_chunks)

            # --- Placeholder: Load embeddings and build index in memory ---
            # Replace this with loading the actual saved FAISS index or ChromaDB client
            print(f"Loading embeddings from placeholder: {embeddings_file}")
            with open(embeddings_file, 'rb') as f_emb:
                embeddings = pickle.load(f_emb)
            # print("Building in-memory FAISS index from loaded embeddings...")
            # dimension = embeddings.shape[1]
            # self.index = faiss.IndexFlatL2(dimension)
            # self.index.add(embeddings.astype(np.float32))
            # print(f"In-memory FAISS index built with {self.index.ntotal} vectors.")
            # For simplicity in placeholder, we'll just store embeddings directly
            self.embeddings = embeddings.astype(np.float32) # Store embeddings
            print("Placeholder: Loaded embeddings directly instead of FAISS index.")
            # --- End Placeholder ---

            print(f"Vector store components loaded ({len(self.text_chunks)} chunks).")
        else:
            print(f"Error: Vector store files not found in {config.VECTOR_STORE_PATH}.")
            print("Please run 'main_build_kb.py' first.")
            raise FileNotFoundError("Vector store not found.")

        print("--- Retriever Initialized ---")

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[str]:
        """
        Embeds the query and retrieves the top_k most similar text chunks.
        """
        if not self.embedding_model or self.text_chunks is None or self.embeddings is None: # Check placeholder embeddings
            print("Error: Retriever not properly initialized.")
            return []

        print(f"\nRetrieving top {top_k} chunks for query: '{query[:100]}...'")

        # 1. Embed the query
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            print(f"Error encoding query: {e}")
            return []

        # 2. Search the index
        try:
            # --- Placeholder Search (Manual Cosine Similarity) ---
            # Replace with self.index.search(query_embedding.astype(np.float32), top_k) for FAISS
            # Calculate cosine similarity between query embedding and all chunk embeddings
            query_emb_norm = np.linalg.norm(query_embedding)
            chunk_embs_norm = np.linalg.norm(self.embeddings, axis=1)
            # Handle potential zero norms
            valid_indices = (query_emb_norm > 1e-9) & (chunk_embs_norm > 1e-9)
            similarities = np.zeros(self.embeddings.shape[0])
            if query_emb_norm > 1e-9:
                 similarities[valid_indices] = np.dot(self.embeddings[valid_indices], query_embedding.T).flatten() / (chunk_embs_norm[valid_indices] * query_emb_norm)

            # Get top k indices (handling cases where k > number of valid similarities)
            num_results = min(top_k, len(similarities))
            # Use argpartition for efficiency if k is small compared to total chunks
            # indices = np.argpartition(-similarities, range(num_results))[:num_results]
            # Sort these top indices by similarity score
            # top_k_indices = indices[np.argsort(-similarities[indices])]

            # Simpler approach for moderate sizes: just sort all similarities
            top_k_indices = np.argsort(-similarities)[:num_results]
            distances = 1.0 - similarities[top_k_indices] # Convert similarity to distance-like score (lower is better)

            # --- End Placeholder Search ---

            print(f"Found indices: {top_k_indices}, Distances: {distances}")

            # 3. Get the corresponding text chunks
            retrieved_chunks = [self.text_chunks[i] for i in top_k_indices]
            print(f"Retrieved {len(retrieved_chunks)} chunks.")
            return retrieved_chunks

        except Exception as e:
            print(f"Error during index search: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    try:
        retriever = VectorRetriever()
        sample_query = "乙酰唑胺片的性状是什么？"
        results = retriever.retrieve(sample_query)
        print(f"\nResults for query: '{sample_query}'")
        for i, chunk in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(chunk)
            print("-" * 20)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Could not run example: {e}")