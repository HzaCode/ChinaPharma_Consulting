# src/evaluators.py
# Contains functions to evaluate the performance of different parts of the system:
# 1. Evaluate the fine-tuned LLM directly on QA data.
# 2. Evaluate the end-to-end RAG pipeline performance.

# Import necessary libraries (install evaluate, rouge_score, bert_score, datasets)
from datasets import load_dataset
from evaluate import load as load_metric # Hugging Face evaluate library
import torch

# Import project modules
from . import config
from .llm_handler import LLMAnswerGenerator # To load the model for evaluation
from .rag_pipeline import RAGPipeline       # To evaluate the full pipeline

# Note: Evaluation can be complex. These are simplified examples.

def evaluate_llm_performance():
    """
    Evaluates the fine-tuned LLM's performance on the QA evaluation dataset
    WITHOUT using RAG retrieval. Measures how well it answers based on fine-tuning.
    """
    print("--- Evaluating Fine-tuned LLM Performance (Direct QA) ---")

    # 1. Load Evaluation Data
    print(f"Loading evaluation QA data from: {config.LLM_FINETUNE_EVAL_DATA}")
    try:
        eval_dataset = load_dataset("json", data_files=config.LLM_FINETUNE_EVAL_DATA, split="train")
        # Limit the number of samples for quicker evaluation if needed
        # eval_dataset = eval_dataset.select(range(50))
        print(f"Loaded {len(eval_dataset)} samples for LLM evaluation.")
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return None

    # 2. Load the Fine-tuned LLM (via LLMAnswerGenerator for convenience)
    try:
        llm_generator = LLMAnswerGenerator()
        # We only need the model and tokenizer part for direct generation
        if not llm_generator.model or not llm_generator.tokenizer:
            raise RuntimeError("LLM Handler failed to load model/tokenizer.")
    except Exception as e:
        print(f"Error initializing LLM Answer Generator for evaluation: {e}")
        return None

    # 3. Load Evaluation Metrics (e.g., ROUGE)
    try:
        rouge_metric = load_metric("rouge")
        # bleu_metric = load_metric("bleu") # Optional
        # bertscore_metric = load_metric("bertscore") # Optional (requires more setup)
        print("Evaluation metrics (ROUGE) loaded.")
    except Exception as e:
        print(f"Error loading evaluation metrics: {e}")
        return None

    # 4. Generate Predictions and Calculate Metrics
    predictions = []
    references = []

    print("Generating predictions for evaluation dataset...")
    # Define a simple prompt template for direct QA (without context)
    def build_direct_qa_prompt(query):
         # Adjust based on the model's expected format
         return f"### Question:\n{query}\n\n### Answer:"

    for example in eval_dataset:
        query = example.get("question")
        reference_answer = example.get("answer")
        if not query or not reference_answer:
            continue

        prompt = build_direct_qa_prompt(query)
        references.append(reference_answer)

        # Generate answer using the loaded fine-tuned model directly
        try:
            # Using pipeline if available
            if llm_generator.generation_pipeline:
                 outputs = llm_generator.generation_pipeline(
                      prompt,
                      max_new_tokens=150, # Shorter length for direct eval
                      num_return_sequences=1,
                      eos_token_id=llm_generator.tokenizer.eos_token_id,
                      pad_token_id=llm_generator.tokenizer.pad_token_id
                 )
                 generated_text = outputs[0]['generated_text']
                 prediction = generated_text.split("回答:")[-1].strip()

            # Using model.generate otherwise
            else:
                 inputs = llm_generator.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(config.DEVICE)
                 outputs = llm_generator.model.generate(
                     **inputs,
                     max_new_tokens=150,
                     eos_token_id=llm_generator.tokenizer.eos_token_id,
                     pad_token_id=llm_generator.tokenizer.pad_token_id,
                 )
                 generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                 prediction = llm_generator.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            predictions.append(prediction)
            # print(f"Q: {query}\nPred: {prediction}\nRef: {reference_answer}\n---") # Uncomment for debug

        except Exception as e:
            print(f"Error generating prediction for query '{query}': {e}")
            predictions.append("") # Add empty prediction on error

    # 5. Compute Metrics
    print("\nComputing evaluation metrics...")
    if predictions and references:
        try:
            rouge_results = rouge_metric.compute(predictions=predictions, references=references)
            print("\n--- Fine-tuned LLM Evaluation Results (ROUGE) ---")
            for key, value in rouge_results.items():
                print(f"{key}: {value*100:.2f}") # Print as percentage

            # Compute other metrics if loaded
            # bleu_results = bleu_metric.compute(...)
            # bertscore_results = bertscore_metric.compute(...)

            return rouge_results # Return the computed scores
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return None
    else:
        print("No predictions or references available to compute metrics.")
        return None


def evaluate_rag_performance():
    """
    Evaluates the end-to-end RAG pipeline performance using the QA evaluation dataset.
    Measures how well the system answers when using retrieval + generation.
    """
    print("--- Evaluating RAG Pipeline Performance ---")

    # 1. Load Evaluation Data (same QA data)
    print(f"Loading evaluation QA data from: {config.LLM_FINETUNE_EVAL_DATA}")
    try:
        eval_dataset = load_dataset("json", data_files=config.LLM_FINETUNE_EVAL_DATA, split="train")
        # eval_dataset = eval_dataset.select(range(50)) # Limit samples if needed
        print(f"Loaded {len(eval_dataset)} samples for RAG evaluation.")
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return None

    # 2. Initialize the full RAG Pipeline
    try:
        rag_system = RAGPipeline()
        if not rag_system.retriever or not rag_system.llm_generator:
             raise RuntimeError("RAG Pipeline failed to initialize components.")
    except Exception as e:
        print(f"Error initializing RAG Pipeline for evaluation: {e}")
        return None

    # 3. Load Evaluation Metrics
    try:
        rouge_metric = load_metric("rouge")
        print("Evaluation metrics (ROUGE) loaded.")
    except Exception as e:
        print(f"Error loading evaluation metrics: {e}")
        return None

    # 4. Generate Predictions using RAG and Calculate Metrics
    predictions = []
    references = []

    print("Generating predictions using RAG pipeline...")
    for example in eval_dataset:
        query = example.get("question")
        reference_answer = example.get("answer")
        if not query or not reference_answer:
            continue

        references.append(reference_answer)

        # Get answer from the RAG system
        try:
            prediction = rag_system.answer_query(query) # This calls retrieve + generate
            predictions.append(prediction)
            # print(f"Q: {query}\nRAG Pred: {prediction}\nRef: {reference_answer}\n---") # Uncomment for debug
        except Exception as e:
            print(f"Error getting RAG prediction for query '{query}': {e}")
            predictions.append("") # Add empty prediction on error


    # 5. Compute Metrics
    print("\nComputing evaluation metrics for RAG...")
    if predictions and references:
        try:
            rouge_results = rouge_metric.compute(predictions=predictions, references=references)
            print("\n--- RAG Pipeline Evaluation Results (ROUGE) ---")
            for key, value in rouge_results.items():
                print(f"{key}: {value*100:.2f}")

            # Compare these results to the direct LLM evaluation results
            return rouge_results
        except Exception as e:
            print(f"Error computing RAG metrics: {e}")
            return None
    else:
        print("No predictions or references available to compute RAG metrics.")
        return None


if __name__ == '__main__':
    print("\nRunning LLM Direct QA Evaluation...")
    llm_results = evaluate_llm_performance()
    # print("\nLLM Results Summary:", llm_results) # Optional summary

    print("\nRunning RAG Pipeline Evaluation...")
    rag_results = evaluate_rag_performance()
    # print("\nRAG Results Summary:", rag_results) # Optional summary

    # Add comparison logic here if needed
    if llm_results and rag_results:
         print("\nComparison (Example ROUGE-L):")
         print(f"  Direct LLM ROUGE-L: {llm_results.get('rougeL', 0)*100:.2f}")
         print(f"  RAG Pipeline ROUGE-L: {rag_results.get('rougeL', 0)*100:.2f}")