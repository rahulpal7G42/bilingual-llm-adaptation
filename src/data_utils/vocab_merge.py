import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class VocabEngine:
    """Advanced engine for merging regional vocabularies into base LLMs without catastrophic interference."""
    def __init__(self, base_path: str, target_path: str):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_path)

    def merge_and_resize(self, model):
        """Merges tokens and performs mean-initialization for new embeddings."""
        new_tokens = set(self.target_tokenizer.get_vocab().keys()) - set(self.base_tokenizer.get_vocab().keys())
        self.base_tokenizer.add_tokens(list(new_tokens))
        
        model.resize_token_embeddings(len(self.base_tokenizer))
        params = model.get_input_embeddings().weight.data
        
        # Advanced: Initialize new embeddings with the mean of existing ones to stabilize training
        mu = params[:-len(new_tokens)].mean(dim=0)
        params[-len(new_tokens):] = mu
        return model, self.base_tokenizer
