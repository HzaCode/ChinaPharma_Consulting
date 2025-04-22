# Pharmacopoeia RAG: LLM Fine-tuning and Retrieval-Augmented Generation

This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for answering questions based on pharmacopoeia documents. It utilizes a fine-tuned Large Language Model (LLM) and a vector knowledge base populated with pharmacopoeia text embeddings.

## Project Overview

The system primarily consists of the following stages:

1.  **Knowledge Base Indexing:** Text chunks from pharmacopoeia documents are converted into vector embeddings using a Sentence Transformer Model (e.g., BGE `bge-small-zh-v1.5`). These embeddings and their corresponding text chunks are stored for retrieval. *(Note: The current implementation uses Python's `pickle` as a placeholder to save embeddings and text chunks; FAISS indexing code exists but is commented out).*
2.  **LLM Fine-tuning:** A base LLM (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques (like LoRA) on a question-answering dataset relevant to the pharmacopoeia domain. This helps the LLM better understand the domain knowledge and generate more accurate and relevant answers.
3.  **Retrieval-Augmented Generation (RAG):**
    - When a user asks a question, it is converted into an embedding vector using the same Sentence Transformer model (`bge-small-zh-v1.5`).
    - A search is performed over the stored embeddings to find the most relevant text chunks (context) from the pharmacopoeia documents based on embedding similarity. *(Note: The current implementation uses manually calculated cosine similarity as a placeholder; FAISS search code exists but is commented out).*
    - The user's question and the retrieved context are combined into a prompt.
    - The fine-tuned LLM generates an answer based on the provided prompt (question + context).

## Key Components and Technologies

-   **Base LLM:** TinyLlama (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) - Used for answer generation, typically after fine-tuning.
-   **Embedding Model:** Sentence Transformer (`bge-small-zh-v1.5`) - Used to create vector representations of text for retrieval.
-   **Fine-tuning:**
    -   `transformers`: Core library for loading and handling LLMs.
    -   `peft`: For Parameter-Efficient Fine-Tuning (e.g., LoRA).
    -   `trl`: For Supervised Fine-tuning (SFTTrainer).
    -   `bitsandbytes`: For 4-bit quantization (optional memory saving during fine-tuning).
    -   `datasets`: For loading the fine-tuning QA dataset (JSON format).
    -   `accelerate`: For optimizing training across hardware.
-   **Vector Store/Retrieval:**
    -   `sentence-transformers`: For generating embeddings.
    -   `numpy`: For embedding operations and similarity calculation (current placeholder implementation).
    -   FAISS (`faiss-cpu` or `faiss-gpu`): The intended vector index library *(currently inactive in the code, using pickle/manual search as placeholders)*.
-   **Core Framework:** PyTorch (`torch`)
-   **Evaluation (Optional):** `evaluate`, `rouge_score`, `bert_score`

## Installation and Setup

1.  **Prerequisites:**
    -   Python 3.8+
    -   pip
    -   Git
    -   (Optional but recommended) NVIDIA GPU with CUDA for accelerated fine-tuning and inference (especially if planning to activate `faiss-gpu` and use `bitsandbytes`).
2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/HzaCode/ChinaPharma_Consulting.git
    cd ChinaPharma_Consulting
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    # venv\Scripts\activate
    ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    -   **Note:** `requirements.txt` installs `faiss-cpu` by default. If you have a compatible GPU and CUDA setup and plan to activate the FAISS code, you might need to install `faiss-gpu`:

        ```bash
        pip uninstall faiss-cpu
        # Consult FAISS documentation for specific CUDA version compatibility
        pip install faiss-gpu
        ```

5.  **Model Download:**
    -   The required Hugging Face models (LLM, Embedding Model) are **not** listed in `requirements.txt`.
    -   The `transformers` and `sentence-transformers` libraries will automatically download and cache these models when they are first needed by a script (e.g., scripts involving model loading). Ensure you have an internet connection during the first run.
6.  **Configuration File:**
    -   After installation, **please check and potentially adjust the paths and settings in the `src/config.py` file**. This file is crucial for the scripts to run correctly. See the **Configuration** section below for details.

## Configuration

All major configurations for this project (file paths, model names, hyperparameters, etc.) are centralized in the **`src/config.py`** file.

**Before running any script**, review and modify `src/config.py` according to your environment and needs. Key settings include:

-   `SOURCE_DATA_PATH`: Path to the source documents (e.g., pharmacopoeia text). Default: `"data/source/pharmacopoeia_2020.txt"`
-   `LLM_FINETUNE_TRAIN_DATA`, `LLM_FINETUNE_EVAL_DATA`: Paths to the LLM fine-tuning datasets (JSON format). Defaults: `"data/llm_finetuning/qa_train.json"`, `"data/llm_finetuning/qa_eval.json"`
-   `VECTOR_STORE_PATH`: Directory to save knowledge base components (embeddings, text chunks). Default: `"knowledge_base/vector_store"`
-   `FINE_TUNED_LLM_PATH`: Path to save/load the fine-tuned LLM (or LoRA adapters). Default: `"models/fine_tuned_llm"`
-   `BASE_LLM_MODEL_NAME`: Hugging Face ID or local path of the base LLM. Default: `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
-   `EMBEDDING_MODEL_NAME`: Hugging Face ID or local path of the embedding model. Default: `"bge-small-zh-v1.5"`
-   `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parameters for text splitting. Defaults: `500`, `50`
-   `TOP_K`: Number of text chunks to retrieve in RAG. Default: `3`
-   LLM fine-tuning parameters (`LLM_FINETUNE_EPOCHS`, `LLM_FINETUNE_BATCH_SIZE`, `LLM_FINETUNE_LR`, `USE_LORA`, LORA settings).
-   `DEVICE`: Compute device ("cuda" or "cpu"). Auto-detected.

## Usage Instructions

**1. Prepare Data:**

-   Ensure your source document (e.g., pharmacopoeia text file) exists at the path specified by `SOURCE_DATA_PATH` in `src/config.py`.
-   Prepare your QA dataset(s) in JSON format (with "question" and "answer" keys). Ensure the paths match `LLM_FINETUNE_TRAIN_DATA` and `LLM_FINETUNE_EVAL_DATA` in `src/config.py`.

**2. Build the Knowledge Base:**

-   Run the knowledge base building script. It reads configurations (source data path, embedding model, output path, chunking settings) from `src/config.py`.

    ```bash
    python main_build_kb.py
    ```

-   **Note:** The current implementation uses `pickle` to save embeddings (`knowledge_base_embeddings.pkl`) and text chunks (`knowledge_base_chunks.pkl`) in the directory specified by `VECTOR_STORE_PATH`. FAISS indexing code exists in `src/build_knowledge_base.py` but is commented out.

**3. Fine-tune the LLM:**

-   Run the fine-tuning script. It uses the settings defined in `src/config.py` (base model, dataset paths, output path, training hyperparameters, LoRA config).

    ```bash
    python main_finetune_llm.py
    ```

-   This will save the fine-tuned model (or LoRA adapters) to the location specified by `FINE_TUNED_LLM_PATH` in `src/config.py`.

**4. Run RAG Queries (Interactive):**

-   Run the main query script to start an interactive command-line session. It loads the knowledge base and the fine-tuned LLM based on the paths specified in `src/config.py`.

    ```bash
    python main_query.py
    ```

-   Enter your questions directly at the prompt. Type 'exit' or 'quit' to end the session.

**5. Evaluation:**

-   Run the evaluation scripts to assess performance. Configurations (evaluation data path, model paths) are read from `src/config.py`.

    ```bash
    # Evaluate the performance of the end-to-end RAG pipeline
    python main_evaluate_rag.py

    # Evaluate the fine-tuned LLM directly on the QA dataset (without RAG retrieval)
    python main_evaluate_llm.py
    ```

-   The scripts will print evaluation metrics (e.g., ROUGE scores) to the console.

## Contributing

Contributions to this project are welcome! Please feel free to contribute by submitting Pull Requests or creating Issues.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).