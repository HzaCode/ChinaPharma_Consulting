# src/data_processing.py
# Functions for loading, cleaning, and splitting the source document (Pharmacopoeia)
# into smaller chunks suitable for building the RAG knowledge base.

# Import potential libraries (you'll need to install them)
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re

def load_document(file_path: str) -> str:
    """Loads text content from a file."""
    print(f"Loading document from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("Document loaded successfully.")
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading document: {e}")
        return ""

def clean_text(text: str) -> str:
    """(Optional) Cleans the text (e.g., remove extra whitespace, headers/footers)."""
    print("Cleaning text (basic example: stripping whitespace)...")
    # Add more sophisticated cleaning logic if needed (e.g., using regex)
    text = text.strip()
    print("Text cleaned.")
    return text

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits the text into overlapping chunks."""
    print(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    if not text:
        print("Warning: Input text is empty, returning no chunks.")
        return []

    # --- Simple Splitting Logic (Example) ---
    # Replace with a more robust splitter like RecursiveCharacterTextSplitter for better results
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): # Prevent infinite loop if overlap >= size
             break
    # --- End Simple Logic ---

    # --- Example using LangChain (Install langchain first) ---
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    # chunks = text_splitter.split_text(text)
    # --- End LangChain Example ---

    print(f"Text split into {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    # Example usage (for testing this script directly)
    import config
    raw_text = load_document(config.SOURCE_DATA_PATH)
    if raw_text:
        cleaned_text = clean_text(raw_text)
        text_chunks = split_text_into_chunks(cleaned_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print("\nFirst 3 chunks (example):")
        for i, chunk in enumerate(text_chunks[:3]):
            print(f"--- Chunk {i+1} ---")
            print(chunk)
            print("-" * 20)