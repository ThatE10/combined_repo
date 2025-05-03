import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import partial
from typing import Tuple

def get_model_and_tokenizer(model_name: str, use_quantizer=False, low_memory_load=False) -> Tuple:
    """Load model and tokenizer with appropriate configuration."""
    print(f"Loading model: {model_name}")
    
    
    # Configure quantization if needed
    bnb_config = BitsAndBytesConfig(load_in_4bit=True) if use_quantizer else None
    
    if low_memory_load:
        # Load the model
        print("Loading model with low memory usage... e.g. inference")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    else:	
        print("Loading model with full precision...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    model = model.to(device=device)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully")

    print(f"Model loaded and moved to {device}")
    return model, tokenizer

# def create_judge_function(model, tokenizer):
#     """Create a function for the LLM judge."""
#     def judge_fn(prompt):
#         inputs = tokenizer(prompt, return_tensors="pt")
#         full_outputs = model.generate(**inputs, max_length=600)
#         outputs = full_outputs[:, inputs['input_ids'].shape[-1]:]  # Exclude input tokens
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response
    
#     return judge_fn
