# -*- coding: utf-8 -*-
import os
# 1. Memory optimization and path configuration
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/user/syt/hf_models"

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from repe import rep_reading_pipeline
from token_loader import load_token

# Set plot style
sns.set_theme(style="whitegrid")

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    print(f"Loading model: {model_name}...")
    token = load_token()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_data(dataset, factor, split):
    """Extract data from json"""
    raw_data = dataset[factor][split]['data']
    # raw_data is [[pos, neg], [pos, neg]...]
    return raw_data

def evaluate_direction(rep_runner, model, tokenizer, direction, test_data, layer_id):
    """
    Manual evaluation: Calculate projection accuracy of test set on the direction vector
    Accuracy = Proportion of (Score_Pos > Score_Neg)
    """
    # test_data is [pos, neg, pos, neg, ...] (flat list)
    flat_data = test_data
    
    # Get Hidden States using the pipeline
    # _batched_string_to_hiddens returns a dict {layer: array}
    # array shape: (n_samples, hidden_dim)
    # We only need the specific layer
    
    # Note: rep_token=-1 means last token
    H_all = rep_runner._batched_string_to_hiddens(
        flat_data, 
        rep_token=-1, 
        hidden_layers=[layer_id], 
        batch_size=32,
        which_hidden_states=None
    )
    
    H_layer = H_all[layer_id] # shape: (n_samples, hidden_dim)
    
    # Reassemble into (n_pairs, 2, hidden_dim)
    H_pos = H_layer[0::2]
    H_neg = H_layer[1::2]
    
    # Convert to tensor for calculation
    direction = torch.tensor(direction, dtype=torch.float16, device=model.device)
    H_pos = torch.tensor(H_pos, dtype=torch.float16, device=model.device)
    H_neg = torch.tensor(H_neg, dtype=torch.float16, device=model.device)
    
    scores_pos = torch.matmul(H_pos, direction)
    scores_neg = torch.matmul(H_neg, direction)
    
    # Calculate Accuracy: Pos Score > Neg Score
    accuracy = (scores_pos > scores_neg).float().mean().item()
    
    return accuracy

def plot_results(results, save_path):
    plt.figure(figsize=(12, 6))
    
    for factor, acc_list in results.items():
        layers = range(len(acc_list))
        plt.plot(layers, acc_list, label=f"{factor}", marker='o', markersize=3)
        
    plt.title("LAT Accuracy across Layers (Llama-3.1-8B)", fontsize=16)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label="Random Chance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_data_dir = os.path.abspath(os.path.join(script_dir, "../../data/jealousy"))
    dataset_path = os.path.join(abs_data_dir, "jealousy_dataset.json")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    model, tokenizer = load_model()
    
    # Initialize Pipeline
    rep_runner = rep_reading_pipeline.RepReadingPipeline(model=model, tokenizer=tokenizer)
    
    factors = ["relevance", "superiority", "clothing"]
    layers = list(range(model.config.num_hidden_layers)) 
    
    all_results = {}
    
    for factor in factors:
        if factor not in dataset:
            print(f"Skipping {factor} (not in dataset)")
            continue
            
        print(f"\nAnalyzing factor: {factor}...")
        train_data = get_data(dataset, factor, "train") # [p, n, p, n...]
        test_data = get_data(dataset, factor, "test")   # [p, n, p, n...]
        
        print(f"  Train data size: {len(train_data)} items")
        
        if len(train_data) % 2 != 0:
            print("Error: Train data length is odd! Truncating last item.")
            train_data = train_data[:-1]
            
        if len(test_data) % 2 != 0:
             test_data = test_data[:-1]
        
        layer_accuracies = []
        
        print("  Training directions...")
        # get_directions returns a RepReader instance (DirectionFinder in concept)
        # We use PCA by default
        rep_reader = rep_runner.get_directions(
            train_data, 
            rep_token=-1, 
            hidden_layers=layers,
            n_difference=1,
            batch_size=32,
            direction_method='pca'
        )
        
        directions = rep_reader.directions
        
        print("  Evaluating on test set...")
        for layer_id in tqdm(layers):
            # directions[layer_id] is shape (n_components, hidden_dim)
            # We use component 0
            direction = directions[layer_id][0]
            
            acc = evaluate_direction(rep_runner, model, tokenizer, direction, test_data, layer_id)
            layer_accuracies.append(acc)
            
        all_results[factor] = layer_accuracies
        
        max_acc = max(layer_accuracies)
        best_layer = layer_accuracies.index(max_acc)
        print(f"  -> Best Layer: {best_layer}, Accuracy: {max_acc:.4f}")

    results_json_path = os.path.join(abs_data_dir, "analysis_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f)
        
    plot_path = os.path.join(abs_data_dir, "jealousy_layers_accuracy.png")
    plot_results(all_results, plot_path)

if __name__ == "__main__":
    main()
