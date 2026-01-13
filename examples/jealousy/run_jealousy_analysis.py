from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import os
import numpy as np
from repe import repe_pipeline_registry

# 注册 RepE pipeline
repe_pipeline_registry()

def load_jealousy_dataset(data_dir, factor):
    path = os.path.join(data_dir, "jealousy_dataset.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集未找到: {path}。请先运行 generate_jealousy_data.py")
        
    with open(path, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    if factor not in full_dataset:
        raise ValueError(f"因子 '{factor}' 不在数据集中。可用因子: {list(full_dataset.keys())}")
        
    return full_dataset[factor]

def main():
    # --- 1. 配置参数 ---
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct" 
    data_dir = "../../data/jealousy"
    target_factor = "relevance" # 可修改为 'superiority' 或 'clothing'
    
    print(f"正在加载模型: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    
    # 修正 Padding Token 设置 (Llama 3 最佳实践)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 加载数据 ---
    print(f"正在加载数据集因子: {target_factor}")
    dataset = load_jealousy_dataset(data_dir, target_factor)
    
    # --- 3. 初始化 RepReading Pipeline ---
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    
    # --- 4. 提取方向 (PCA) ---
    # 扫描模型最后几层
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    direction_method = 'pca'
    
    print("开始训练 RepReader (PCA)...")
    jealousy_rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=-1,   # 使用最后一个 token 的表示
        hidden_layers=hidden_layers, 
        n_difference=1, # 使用成对差异 (High - Low)
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        batch_size=8,
    )
    
    print("训练完成。正在测试集上评估...")
    
    # --- 5. 评估 / 验证 ---
    # 计算测试数据在提取方向上的投影分数
    H_tests = rep_reading_pipeline(
        dataset['test']['data'], 
        rep_token=-1, 
        hidden_layers=hidden_layers, 
        rep_reader=jealousy_rep_reader,
        batch_size=8
    )
    
    # H_tests 包含了每一层的投影分数
    # 你可以在这里添加代码来计算准确率 (即 High 的分数是否 > Low 的分数)
    
    print(f"因子 {target_factor} 的分析已完成。方向已提取。")
    # 后续可以添加代码保存 direction 或进行可视化

if __name__ == "__main__":
    main()
