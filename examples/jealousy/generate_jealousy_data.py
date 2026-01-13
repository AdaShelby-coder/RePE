# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 定义每个因子的 Prompt 模板 (参考 data/jealousy/LATprompt 文件)
# 修改 TEMPLATES 定义，只保留核心 Content，移除硬编码的 User/Assistant
TEMPLATES = {
    "relevance": "Evaluate the importance of the domain to the narrator in the following scenario. Is the importance 'High' or 'Low'?:\nScenario: {scenario}",
    "superiority": "Evaluate the other person's advantage over the narrator. Is it 'High' or 'Low'?\nScenario: {scenario}",
    "clothing": "Is the color tone of the main object 'Beige' or 'Grey'?\nScenario: {scenario}"
}

# 对应的 Assistant 引导词 (Suffix)
SUFFIXES = {
    "relevance": "The level of importance is",
    "superiority": "The level of advantage is",
    "clothing": "The tone of the main object is"
}

# 定义每个因子的正负标签词
TARGET_WORDS = {
    "relevance": {"pos": "High", "neg": "Low"},
    "superiority": {"pos": "High", "neg": "Low"},
    "clothing": {"pos": "Beige", "neg": "Grey"}
}

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    # Set HF_ENDPOINT to mirror site for faster download
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer.pad_token_id = 0
    return model, tokenizer



def filter_data_by_model(model, tokenizer, pair, factor_name):
    """
    Check if the model correctly predicts this pair of data
    pair: [pos_text, neg_text] (corresponding to Label 1 and Label 0)
    factor_name: factor name
    """
    pos_text, neg_text = pair
    targets = TARGET_WORDS[factor_name]
    pos_word = targets["pos"]
    neg_word = targets["neg"]
    
    # 检查正样本 (High/Beige)
    # 我们比较 P(pos_word) 和 P(neg_word)
    # 对于 Mistral 这样的 Instruction 模型，通常预测概率最高的那个词就是它的答案
    
    # 获取正样本的 logits
    inputs_pos = tokenizer(pos_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_pos = model(**inputs_pos).logits[0, -1, :]
    
    # 获取负样本的 logits
    inputs_neg = tokenizer(neg_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_neg = model(**inputs_neg).logits[0, -1, :]
        
    # 获取 token ids
    pos_id = tokenizer.encode(" " + pos_word, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(" " + neg_word, add_special_tokens=False)[0]
    
    # Criteria:
    # 1. Positive samples tend to output pos_word
    # 2. Negative samples tend to output neg_word
    
    probs_pos = F.softmax(logits_pos, dim=-1)
    probs_neg = F.softmax(logits_neg, dim=-1)
    
    score_pos_pos = probs_pos[pos_id].item()
    score_pos_neg = probs_pos[neg_id].item()
    
    score_neg_pos = probs_neg[pos_id].item()
    score_neg_neg = probs_neg[neg_id].item()
    
    # Only keep this pair if the model predicts correctly on both samples
    is_pos_correct = score_pos_pos > score_pos_neg
    is_neg_correct = score_neg_neg > score_neg_pos
    
    return is_pos_correct and is_neg_correct

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
        
        # 应用模板
        if template:
            text1 = template.format(scenario=text1)
            text2 = template.format(scenario=text2)
        
        label1 = int(row1['Label'])
        label2 = int(row2['Label'])
        
        # 构建配对: [正样本, 负样本]
        if label1 == 1 and label2 == 0:
            pair = [text1, text2]
            current_labels = [1, 0]
        elif label1 == 0 and label2 == 1:
            pair = [text2, text1]
            current_labels = [1, 0] 
        else:
            # print(f"Warning: Row {i} and {i+1} are not a 1-0 pair. Skipping.")
            continue
            
        total_pairs += 1
        
        # 模型验证
        if model and tokenizer:
            if not filter_data_by_model(model, tokenizer, pair, factor_name):
                # 模型判断错误，跳过该对
                continue
        
        valid_count += 1
        data.extend(pair)
        labels.append(current_labels)
        
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

    # 初始化模型 (如果不需要过滤，可以将 use_model 设为 False)
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
    dataset = {}
    
    for factor in factors:
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
        
    output_path = os.path.join(data_dir, "jealousy_dataset.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully saved dataset to: {output_path}")

if __name__ == "__main__":
    main()
