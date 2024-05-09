# Import the necessary classes and functions
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer)
import torch

def init_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    # Check if MPS is available and set the device accordingly
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize the model and tokenizer with MPS or CPU as the device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)  # Move model to the appropriate device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set for causal LM
    # Adjust tokenizer settings if necessary
    tokenizer.padding_side = "right"

    return model, tokenizer

