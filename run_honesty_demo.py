import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import numpy as np
import pandas as pd
import random
from repe import repe_pipeline_registry

# Register the pipeline
repe_pipeline_registry()

# Configuration
model_name_or_path = "gpt2" # Using gpt2 for faster testing, though performance might be poor compared to larger models
data_path = "data/facts/facts_true_false.csv"

print(f"Loading model: {model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = 0

# --- Data Preparation Function (simplified from utils.py) ---
def honesty_function_dataset(data_path: str, tokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> dict:
    random.seed(seed)
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Using a small subset for quick testing
    limit = 10 
    for statement in true_statements[:limit]:
        honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + statement)
        untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + statement)

    ntrain = 5
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    test_data = np.concatenate(combined_data[ntrain:]).tolist()
    
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * (len(test_data)//2)}
    }

# --- Main Logic ---
print("Preparing data...")
user_tag = ""
assistant_tag = ""
dataset = honesty_function_dataset(data_path, tokenizer, user_tag, assistant_tag)

print("Initializing RepReading Pipeline...")
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

# Use fewer layers for testing if needed, or all layers
# hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1)) 
hidden_layers = [5] # Just testing one layer for speed

print(f"Training RepReader on layers: {hidden_layers}")
direction_method = 'pca'
rep_reader = rep_reading_pipeline.get_directions(
    dataset['train']['data'], 
    rep_token=-1, 
    hidden_layers=hidden_layers, 
    n_difference=1, 
    train_labels=dataset['train']['labels'], 
    direction_method=direction_method,
)

print("RepReader trained successfully!")

print("Running inference on test data...")
# Simple test on a few examples
test_input = [
    "Pretend you're an honest person making statements about the world. The earth is round.",
    "Pretend you're an untruthful person making statements about the world. The earth is flat."
]

results = rep_reading_pipeline(
    test_input, 
    rep_token=-1, 
    hidden_layers=hidden_layers, 
    rep_reader=rep_reader,
    batch_size=2
)

print("\nResults:")
for i, text in enumerate(test_input):
    print(f"\nInput: {text}")
    # results is a dict mapping layer to scores
    # results[layer] is a list of lists if batch_size > 1 ? Let's print structure to debug
    # Actually rep_reading_pipeline returns a list of results (one per input) if passed a list
    # But here we are using a custom pipeline, let's see how it behaves.
    # Usually pipelines return a list of dicts.
    
    # Based on the error "IndexError: list index out of range", maybe results is a list of results
    pass

# Debug print
# print(f"Results type: {type(results)}")
# print(f"Results: {results}")

for i, text in enumerate(test_input):
    print(f"\nInput: {text}")
    for layer in hidden_layers:
        # Check if results is list of dicts or dict of lists
        if isinstance(results, list):
             # Assuming list of dicts, where each dict has keys as layers?
             # Or maybe list of results corresponding to inputs
             if i < len(results):
                 res = results[i]
                 # rep-reading pipeline structure might be specific
                 # Let's assume res contains scores for requested layers
                 if layer in res:
                     score = res[layer]
                     print(f"Layer {layer} Honesty Score: {score}")
                 else:
                     # It is possible the structure is different
                     # The original code assumed results[layer][i] which suggests results is a dict {layer: [scores]}
                     # But pipeline typically returns [ {result_for_input_1}, {result_for_input_2} ]
                     pass
        elif isinstance(results, dict):
             if layer in results:
                 if i < len(results[layer]):
                     score = results[layer][i]
                     print(f"Layer {layer} Honesty Score: {score}")


print("\nDone!")
