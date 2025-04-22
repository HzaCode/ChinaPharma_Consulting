# src/build_knowledge_base.py
# Uses the pre-trained EMBEDDING MODEL to create vector embeddings
# for the text chunks and stores them in a vector database (e.g., FAISS, ChromaDB).

# Import necessary libraries (install sentence-transformers, faiss-cpu/gpu or chromadb)
from sentence_transformers import SentenceTransformer
# import faiss
# import chromadb
import numpy as np
import os
import pickle # Using pickle for simple saving with FAISS index

# Import project modules
from . import config
from . import data_processing

def build_and_save_vector_store():
    """
    Loads data, generates embeddings using the configured embedding model,
    builds a vector index, and saves it along with the corresponding text chunks.
    """
    print("--- Building Knowledge Base ---")

    # 1. Load and process document
    raw_text = data_processing.load_document(config.SOURCE_DATA_PATH)
    if not raw_text:
        print("Halting knowledge base build due to missing source data.")
        return
    cleaned_text = data_processing.clean_text(raw_text)
    text_chunks = data_processing.split_text_into_chunks(
        cleaned_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP
    )
    if not text_chunks:
        print("Halting knowledge base build due to no text chunks generated.")
        return

    # 2. Load Embedding Model
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    try:
        # Use trust_remote_code=True if needed for models like bge
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE, trust_remote_code=True)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return

    # 3. Generate Embeddings
    print(f"Generating embeddings for {len(text_chunks)} chunks (this may take a while)...")
    try:
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
        print(f"Embeddings generated successfully. Shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # 4. Build and Save Vector Store (Example using simple FAISS index + pickle)
    #    For ChromaDB or others, the saving mechanism will differ.
    output_dir = config.VECTOR_STORE_PATH
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "knowledge_base.index") # FAISS index file
    chunks_file = os.path.join(output_dir, "knowledge_base_chunks.pkl") # Corresponding text chunks

    print(f"Building FAISS index...")
    try:
        dimension = embeddings.shape[1]
        # Using IndexFlatL2, simple but effective for moderate size. Use others for large datasets.
        # index = faiss.IndexFlatL2(dimension)
        # index.add(embeddings.astype(np.float32)) # FAISS requires float32
        # print(f"FAISS index built with {index.ntotal} vectors.")

        # Save the index and the chunks
        # faiss.write_index(index, index_file)
        with open(chunks_file, 'wb') as f_chunks:
            pickle.dump(text_chunks, f_chunks)

        # --- Placeholder ---
        # Since FAISS might not be installed, let's just save embeddings and chunks directly for now
        # You should replace this with actual FAISS or ChromaDB saving
        embeddings_file = os.path.join(output_dir, "knowledge_base_embeddings.pkl")
        with open(embeddings_file, 'wb') as f_emb:
             pickle.dump(embeddings, f_emb)
        print(f"FAISS index saved to: {index_file} (Placeholder: embeddings saved)")
        print(f"Text chunks saved to: {chunks_file}")
        # --- End Placeholder ---

    except Exception as e:
        print(f"Error building or saving FAISS index/chunks: {e}")
        # Consider cleaning up partially created files if needed

    print("--- Knowledge Base Build Finished ---")


if __name__ == '__main__':
    build_and_save_vector_store()