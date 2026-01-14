# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import os
import numpy as np
from repe import repe_pipeline_registry
from token_loader import load_token

# зЂВс RepE pipeline
repe_pipeline_registry()

def load_jealousy_dataset(data_dir, factor):
    # Absolute path to the data directory
    abs_data_dir = "/home/user/syt/representation-engineering/data/jealousy"
    path = os.path.join(abs_data_dir, "jealousy_dataset.json")
    
    if not os.path.exists(path):
        # Try relative path
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/jealousy/jealousy_dataset.json"))
        if not os.path.exists(path):
             raise FileNotFoundError(f"Dataset not found: {path}. Please run generate_jealousy_data.py first")
        
    with open(path, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    if factor not in full_dataset:
        raise ValueError(f"Factor '{factor}' not in dataset. Available factors: {list(full_dataset.keys())}")
        
    return full_dataset[factor]

def main():
    # --- 1. Configuration ---
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct" 
    data_dir = "../../data/jealousy"
    target_factor = "relevance" # Can be changed to 'superiority' or 'clothing'
    token = load_token()
    
    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=token)
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=token)
    
    # Fix Padding Token setting (Llama 3 Best Practice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Data ---
    print(f"Loading dataset factor: {target_factor}")
    dataset = load_jealousy_dataset(data_dir, target_factor)
    
    # --- 3. Initialize RepReading Pipeline ---
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    
    # --- 4. Extract Directions (PCA) ---
    # Scan last few layers
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    direction_method = 'pca'
    
    print("Start training RepReader (PCA)...")
    jealousy_rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=-1,   # Use last token representation
        hidden_layers=hidden_layers, 
        n_difference=1, # Use pair difference (High - Low)
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        batch_size=8,
    )
    
    print("Training complete. Evaluating on test set...")
    
    # --- 5. Evaluate / Validate ---
    # Calculate projection scores of test data on extracted direction
    H_tests = rep_reading_pipeline(
        dataset['test']['data'], 
        rep_token=-1, 
        hidden_layers=hidden_layers, 
        rep_reader=jealousy_rep_reader,
        batch_size=8
    )
    
    # H_tests contains projection scores for each layer
    # You can add code here to calculate accuracy (i.e. whether High score > Low score)
    
    print(f"Analysis for factor {target_factor} completed. Direction extracted.")
    # Code can be added to save direction or visualize

if __name__ == "__main__":
    main()
