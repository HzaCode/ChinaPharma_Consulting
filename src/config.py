# src/config.py
# Central configuration file for the project.
# Stores paths, model names, hyperparameters, etc.

import torch

# --- Paths ---
SOURCE_DATA_PATH = "data/source/pharmacopoeia_2020.txt" # Path to the raw Pharmacopoeia text
LLM_FINETUNE_TRAIN_DATA = "data/llm_finetuning/qa_train.json" # Path to QA training data for LLM
LLM_FINETUNE_EVAL_DATA = "data/llm_finetuning/qa_eval.json"   # Path to QA evaluation data for LLM
VECTOR_STORE_PATH = "knowledge_base/vector_store"             # Directory to save the vector index
BASE_LLM_MODEL_PATH = "models/base_llm"                       # (Optional) Local path for base LLM
FINE_TUNED_LLM_PATH = "models/fine_tuned_llm"                 # Path to save/load fine-tuned LLM (or LoRA adapter)
EMBEDDING_MODEL_PATH = "models/embedding_model"               # (Optional) Local path for embedding model

# --- Model Names ---
# Use model IDs from Hugging Face Hub or local paths
BASE_LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # Base LLM to fine-tune
EMBEDDING_MODEL_NAME = "bge-small-zh-v1.5"                    # Embedding model for RAG retrieval (Sentence Transformer)

# --- Data Processing & RAG ---
CHUNK_SIZE = 500               # Target size for text chunks when building the knowledge base
CHUNK_OVERLAP = 50             # Overlap between consecutive chunks
TOP_K = 3                      # Number of relevant chunks to retrieve for RAG

# --- LLM Fine-tuning ---
# Example parameters (adjust based on resources and experimentation)
LLM_FINETUNE_EPOCHS = 1        # Number of training epochs for LLM fine-tuning
LLM_FINETUNE_BATCH_SIZE = 4    # Batch size for LLM fine-tuning (adjust based on GPU memory)
LLM_FINETUNE_LR = 2e-5         # Learning rate for LLM fine-tuning
USE_LORA = True                # Whether to use LoRA/PEFT for efficient fine-tuning
LORA_R = 8                     # LoRA rank
LORA_ALPHA = 16                # LoRA alpha
LORA_DROPOUT = 0.05            # LoRA dropout
# Add other training arguments as needed (gradient accumulation, warmup steps, etc.)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Set device for computations (GPU if available)

print(f"Configuration loaded. Using device: {DEVICE}")
# Add any other configuration variables needed across the project