# -*- coding: utf-8 -*-
import os
# Set HF_ENDPOINT to mirror site for faster download - MUST be done before importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Set HF_HOME to a custom directory on NVMe SSD for better performance
os.environ["HF_HOME"] = "/home/user/syt/hf_models"

import pandas as pd
import json
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from token_loader import load_token

# Define Prompt Templates for each factor (Reference: data/jealousy/LATprompt)
# Modified TEMPLATES to keep only core content, removed hardcoded User/Assistant
TEMPLATES = {
    "relevance": "Evaluate the importance of the domain to the narrator in the following scenario. Is the importance 'High' or 'Low'?:\nScenario: {scenario}",
    "superiority": "Evaluate the other person's advantage over the narrator. Is it 'High' or 'Low'?\nScenario: {scenario}",
    "clothing": "Is the color tone of the main object 'Beige' or 'Grey'?\nScenario: {scenario}"
}

# Corresponding Assistant Suffixes
SUFFIXES = {
    "relevance": "The level of importance is",
    "superiority": "The level of advantage is",
    "clothing": "The tone of the main object is"
}

# Define positive/negative label words for each factor
TARGET_WORDS = {
    "relevance": {"pos": "High", "neg": "Low"},
    "superiority": {"pos": "High", "neg": "Low"},
    "clothing": {"pos": "Beige", "neg": "Grey"}
}

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    print(f"Loading model: {model_name}")
    # Read token from file or env
    token = load_token()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token=token)
    
    # Fix Padding Token setting
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer



def format_prompt_llama3(tokenizer, template, scenario, suffix):
    """
    Format prompt specifically for Llama-3
    """
    # 1. Build User Message
    messages = [{"role": "user", "content": template.format(scenario=scenario)}]
    
    # 2. Apply Chat Template
    # add_generation_prompt=True adds <|start_header_id|>assistant<|end_header_id|>
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 3. Append Suffix (guide model output)
    return prompt + suffix

def filter_data_by_model(model, tokenizer, pos_prompt, neg_prompt, factor_name):
    """
    Verify if the model can correctly distinguish positive/negative samples
    """
    targets = TARGET_WORDS[factor_name]
    pos_word = targets["pos"]
    neg_word = targets["neg"]
    
    # Note: Llama3 Tokenizer is sensitive to spaces, usually need to add space before word
    pos_id = tokenizer.encode(" " + pos_word, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(" " + neg_word, add_special_tokens=False)[0]
    
    def get_probs(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        return F.softmax(logits, dim=-1)

    probs_pos_input = get_probs(pos_prompt)
    probs_neg_input = get_probs(neg_prompt)
    
    # Logic:
    # Positive input -> Prob(Pos Word) > Prob(Neg Word)
    # Negative input -> Prob(Neg Word) > Prob(Pos Word)
    cond1 = probs_pos_input[pos_id].item() > probs_pos_input[neg_id].item()
    cond2 = probs_neg_input[neg_id].item() > probs_neg_input[pos_id].item()
    
    return cond1 and cond2

def process_csv(file_path, factor_name, model=None, tokenizer=None):
    print(f"Reading {file_path}")
    try:
        # First try GBK
        df = pd.read_csv(file_path, encoding='gbk')
    except Exception:
        try:
            # Fallback to UTF-8
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return [], []
        
    data = []
    labels = []
    
    template = TEMPLATES.get(factor_name)
    suffix = SUFFIXES.get(factor_name)
    if not template:
        print(f"Warning: Template for factor {factor_name} not found, using raw text.")
    
    valid_count = 0
    total_pairs = 0
    
    # 假设 CSV 是成对排列的 (每一对是一个实验组)
    for i in tqdm(range(0, len(df), 2), desc=f"Processing {factor_name}"):
        if i+1 >= len(df):
            break
            
        row1 = df.iloc[i]
        row2 = df.iloc[i+1]
        
        text1 = str(row1['English Text']).strip()
        text2 = str(row2['English Text']).strip()
        
        label1 = int(row1['Label'])
        label2 = int(row2['Label'])
        
        # Determine positive/negative order
        if label1 == 1 and label2 == 0:
            pos_text, neg_text = text1, text2
        elif label1 == 0 and label2 == 1:
            pos_text, neg_text = text2, text1
        else:
            continue
            
        # --- Format Prompt ---
        if tokenizer:
            pos_prompt = format_prompt_llama3(tokenizer, template, pos_text, suffix)
            neg_prompt = format_prompt_llama3(tokenizer, template, neg_text, suffix)
        else:
            # For debugging only
            pos_prompt = f"User: {pos_text}\nAssistant: {suffix}"
            neg_prompt = f"User: {neg_text}\nAssistant: {suffix}"
            
        total_pairs += 1
        
        # Model Validation
        if model and tokenizer:
            if not filter_data_by_model(model, tokenizer, pos_prompt, neg_prompt, factor_name):
                # Skip if model predicts incorrectly
                continue
        
        valid_count += 1
        
        # --- Critical: Store as flat list, keep [Pos, Neg] order ---
        data.append(pos_prompt)
        data.append(neg_prompt)
        labels.append(1)
        labels.append(0)
        
    if model:
        print(f"  {factor_name}: Kept {valid_count}/{total_pairs} pairs (filtered out model mispredictions)")
        
    return data, labels

def main():
    base_data_path = "../../data/jealousy"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, base_data_path))
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # Initialize model (set use_model=False if filtering is not needed)
    use_model = True 
    model = None
    tokenizer = None
    
    if use_model:
        try:
            model, tokenizer = load_model()
        except Exception as e:
            print(f"Model load failed: {e}")
            print("Skipping model verification step, generating data directly.")
            use_model = False

    factors = ["relevance", "superiority", "clothing"]
    
    # 1. Try to load existing dataset for resume capability
    output_path = os.path.join(data_dir, "jealousy_dataset.json")
    if os.path.exists(output_path):
        print(f"Found existing dataset, attempting to resume: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except:
            print("Load failed, creating new dataset")
            dataset = {}
    else:
        dataset = {}
    
    for factor in factors:
        # 2. Skip if factor already processed
        if factor in dataset and len(dataset[factor]['train']['data']) > 0:
            print(f"Factor {factor} already exists in dataset, skipping.")
            continue

        csv_path = os.path.join(data_dir, f"{factor}.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
            
        print(f"\nProcessing factor: {factor}...")
        data, labels = process_csv(csv_path, factor, model, tokenizer)
        
        if not data:
            print(f"  {factor}: No valid data pairs")
            continue

        n_pairs = len(labels)
        n_train = int(n_pairs * 0.8)
        
        dataset[factor] = {
            "train": {
                "data": data[:n_train*2],
                "labels": labels[:n_train]
            },
            "test": {
                "data": data[n_train*2:],
                "labels": labels[n_train:]
            }
        }
        print(f"  - {factor}: {n_train} training pairs, {n_pairs - n_train} test pairs")
        
        # 3. Save checkpoint after each factor
        print(f"Saving checkpoint for {factor}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print("Saved.")
        
    print(f"\nSuccessfully saved dataset to: {output_path}")

if __name__ == "__main__":
    main()
