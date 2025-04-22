# src/llm_handler.py
# Handles loading the fine-tuned LLM and tokenizer, constructing the prompt
# with retrieved context, and generating the final answer.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel # Needed if using LoRA

# Import project modules
from . import config

class LLMAnswerGenerator:
    """
    Loads the fine-tuned LLM and generates answers based on a query
    and provided context chunks.
    """
    def __init__(self):
        """Initializes the generator by loading the model and tokenizer."""
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()
        # Optional: Use Hugging Face pipeline for easier generation
        self.generation_pipeline = None
        try:
             if self.model and self.tokenizer:
                  self.generation_pipeline = pipeline(
                       "text-generation",
                       model=self.model,
                       tokenizer=self.tokenizer,
                       device=0 if config.DEVICE == "cuda" else -1 # pipeline uses device index
                  )
                  print("Text generation pipeline created.")
        except Exception as e:
             print(f"Could not create text generation pipeline: {e}")


    def _load_model_and_tokenizer(self):
        """Loads the fine-tuned LLM and its tokenizer."""
        print("--- Initializing LLM Answer Generator ---")
        model_path = config.FINE_TUNED_LLM_PATH

        print(f"Loading fine-tuned Tokenizer from: {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token # Ensure pad token is set
            print("Tokenizer loaded.")
        except Exception as e:
            print(f"Error loading tokenizer from {model_path}: {e}")
            raise RuntimeError("Failed to load fine-tuned tokenizer.") from e

        print(f"Loading fine-tuned Model from: {model_path}...")
        try:
            # Check if it's a LoRA adapter or a full model save
            # A simple check (might need refinement): look for 'adapter_config.json'
            is_lora_adapter = os.path.exists(os.path.join(model_path, 'adapter_config.json'))

            if is_lora_adapter and config.USE_LORA:
                print("Detected LoRA adapter. Loading base model first...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.BASE_LLM_MODEL_NAME,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32
                )
                print(f"Loading LoRA adapter from {model_path}...")
                self.model = PeftModel.from_pretrained(base_model, model_path)
                # Optional: Merge LoRA weights for faster inference (uses more memory)
                # print("Merging LoRA adapter...")
                # self.model = self.model.merge_and_unload()
                print("LoRA adapter loaded onto base model.")
            else:
                print("Loading full fine-tuned model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32
                )
                print("Full fine-tuned model loaded.")

        except Exception as e:
            print(f"Error loading fine-tuned model from {model_path}: {e}")
            # Fallback or specific error handling can be added here
            raise RuntimeError("Failed to load fine-tuned LLM.") from e

        print("--- LLM Answer Generator Initialized ---")

    def _build_prompt(self, query: str, context_chunks: list[str]) -> str:
        """Builds the prompt string for the LLM using a template."""
        context = "\n\n".join(context_chunks)

        # --- Prompt Template Example ---
        # Adjust this based on how TinyLlama-Chat expects context and questions
        # This template explicitly tells the model to use the provided context.
        template = f"""基于以下信息回答问题。如果信息不足或与问题无关，请说明无法根据提供的信息回答。

[参考信息]
{context}
[/参考信息]

问题: {query}

回答:"""
        # --- End Prompt Template ---

        # --- Alternative Chat Template (if using tokenizer.apply_chat_template) ---
        # messages = [
        #     {"role": "system", "content": "你是一个基于提供的《中国药典》信息回答问题的助手。"},
        #     {"role": "user", "content": f"请根据以下信息回答问题。\n\n[参考信息]\n{context}\n[/参考信息]\n\n问题: {query}"}
        # ]
        # try:
        #     # Note: Not all tokenizers have chat templates configured well. Test this.
        #     prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #     return prompt
        # except Exception as e:
        #     print(f"Warning: Could not apply chat template ({e}). Falling back to basic template.")
        #     return template # Fallback to the basic template
        # --- End Chat Template ---

        return template

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Generates an answer using the loaded LLM and provided context."""
        if not self.model or not self.tokenizer:
            print("Error: LLM model/tokenizer not loaded.")
            return "错误：模型未加载。"

        prompt = self._build_prompt(query, context_chunks)
        print("\n--- Generated Prompt ---")
        print(prompt)
        print("--- End Prompt ---")

        # Generate answer using the model directly or the pipeline
        print("Generating answer...")
        try:
            # --- Using Pipeline (Simpler) ---
            if self.generation_pipeline:
                 # Adjust parameters as needed (max_new_tokens, temperature, etc.)
                 # Need to extract only the generated part, not the prompt
                 outputs = self.generation_pipeline(
                      prompt,
                      max_new_tokens=250, # Limit response length
                      num_return_sequences=1,
                      eos_token_id=self.tokenizer.eos_token_id,
                      pad_token_id=self.tokenizer.pad_token_id, # Set pad token
                      do_sample=True, # Use sampling for potentially more natural answers
                      temperature=0.7,
                      top_p=0.9
                 )
                 generated_text = outputs[0]['generated_text']
                 # Extract only the answer part (logic depends on the prompt template)
                 answer = generated_text.split("回答:")[-1].strip()
                 print("Answer generated via pipeline.")
                 return answer
            # --- Using Model Directly (More control) ---
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(config.DEVICE)
                # Adjust generation parameters as needed
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                # Decode only the newly generated tokens
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                print("Answer generated via model.generate.")
                return answer

        except Exception as e:
            print(f"Error during answer generation: {e}")
            return "错误：生成答案时出错。"


if __name__ == '__main__':
    # Example usage
    try:
        llm_gen = LLMAnswerGenerator()
        sample_query = "乙酰唑胺片的鉴别方法是什么？"
        sample_context = [
            "【鉴别】（1）取本品细粉适量（约相当于乙酰唑胺0.2g），加水3ml与氢氧化钠试液1ml，搅拌，滤过；取滤液2ml，加水8ml摇匀后，照乙酰唑胺项下的鉴别（1）项试验，显相同的反应。",
            "（2）取本品细粉适量（约相当于乙酰唑胺50mg），照乙酰唑胺项下的鉴别（2）项试验，显相同的反应。"
        ]
        answer = llm_gen.generate(sample_query, sample_context)
        print(f"\nQuery: {sample_query}")
        print(f"Generated Answer:\n{answer}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Could not run example: {e}")