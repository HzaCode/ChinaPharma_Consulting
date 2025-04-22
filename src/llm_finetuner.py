# src/llm_finetuner.py
# Handles the fine-tuning process for the base LLM (TinyLlama)
# using the provided QA dataset. Uses PEFT/LoRA for efficiency if configured.

# Import necessary libraries (install transformers, peft, trl, datasets, accelerate, bitsandbytes)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig # For 4-bit quantization (optional)
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # For LoRA
from trl import SFTTrainer # Simplifies supervised fine-tuning

# Import project modules
from . import config

def format_prompt(example):
    """Creates a prompt suitable for the chat model based on the QA pair."""
    # Adjust this template based on the specific chat format TinyLlama expects
    # This is a generic example, might need <|im_start|>, <s>, [INST], etc.
    question = example.get("question", "")
    answer = example.get("answer", "")
    if question and answer:
         # Example format - verify with TinyLlama documentation/community
        return f"### Question:\n{question}\n\n### Answer:\n{answer}"
    else:
        return "" # Skip incomplete examples


def fine_tune_llm():
    """Loads data, configures and runs the LLM fine-tuning process."""
    print("--- Starting LLM Fine-tuning ---")

    # 1. Load Dataset
    print("Loading QA datasets...")
    try:
        train_dataset = load_dataset("json", data_files=config.LLM_FINETUNE_TRAIN_DATA, split="train")
        eval_dataset = load_dataset("json", data_files=config.LLM_FINETUNE_EVAL_DATA, split="train") # Load eval data
        print(f"Datasets loaded. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # (Optional) Add data formatting/preprocessing if needed here before tokenization

    # 2. Load Tokenizer and Model
    print(f"Loading base model and tokenizer: {config.BASE_LLM_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL_NAME, trust_remote_code=True)
        # Set padding token if it doesn't exist (common practice)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token.")

        # Optional: Load model with quantization for memory saving (e.g., 4-bit)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
        # )
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_LLM_MODEL_NAME,
            # quantization_config=bnb_config, # Uncomment for quantization
            device_map="auto", # Automatically distribute across GPUs if available
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32 # Use bfloat16 on GPU if supported
        )
        model.config.use_cache = False # Recommended for fine-tuning
        print("Base model and tokenizer loaded.")

        # Prepare model for k-bit training if using quantization + LoRA
        # if config.USE_LORA and hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        #     print("Preparing model for k-bit training...")
        #     model = prepare_model_for_kbit_training(model)

    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    # 3. Configure PEFT/LoRA (if enabled)
    if config.USE_LORA:
        print("Configuring LoRA...")
        peft_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules = [...] # Specify target modules or let peft infer
        )
        model = get_peft_model(model, peft_config)
        print("LoRA configured. Trainable parameters:")
        model.print_trainable_parameters()
    else:
        print("LoRA not enabled. Performing full fine-tuning (requires more resources).")


    # 4. Configure Training Arguments
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=config.FINE_TUNED_LLM_PATH,
        per_device_train_batch_size=config.LLM_FINETUNE_BATCH_SIZE,
        per_device_eval_batch_size=config.LLM_FINETUNE_BATCH_SIZE,
        num_train_epochs=config.LLM_FINETUNE_EPOCHS,
        learning_rate=config.LLM_FINETUNE_LR,
        logging_dir='./logs',
        logging_steps=50, # Log training progress periodically
        save_strategy="epoch", # Save model checkpoint at the end of each epoch
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        load_best_model_at_end=True, # Load the best performing model based on evaluation
        metric_for_best_model="eval_loss", # Use evaluation loss to determine the best model
        greater_is_better=False, # Lower eval loss is better
        fp16=True if config.DEVICE == "cuda" else False, # Use mixed precision on GPU
        # bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False, # Use bf16 if available
        gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch size
        # Add other arguments like warmup_steps, weight_decay, etc.
        report_to="none" # Disable wandb/tensorboard reporting for simplicity
    )

    # 5. Initialize Trainer (using SFTTrainer for simplicity)
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=format_prompt, # Use the function to format prompts
        max_seq_length=512, # Adjust based on model and data
        # packing=True, # packs multiple short examples into one sequence (efficiency)
        peft_config=peft_config if config.USE_LORA else None,
    )

    # 6. Start Training
    print("Starting training...")
    try:
        trainer.train()
        print("Training finished.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 7. Save the final model (or adapter if using LoRA)
    print(f"Saving final model/adapter to: {config.FINE_TUNED_LLM_PATH}")
    # Trainer saves automatically based on save_strategy, but explicit save is good practice
    # If using LoRA, this saves the adapter; otherwise, saves the full model
    trainer.save_model(config.FINE_TUNED_LLM_PATH)
    # Also save the tokenizer
    tokenizer.save_pretrained(config.FINE_TUNED_LLM_PATH)
    print("Model and tokenizer saved.")

    print("--- LLM Fine-tuning Finished ---")

if __name__ == '__main__':
    fine_tune_llm()