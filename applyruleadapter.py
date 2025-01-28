from datasets import load_dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import sys
import time
import argparse
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import torch
from rules.rule_pool import rule_pool
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
import os
import json
import random



# Load the dataset
print("\n\n==========Load dataset.=============")
# train_dataset = load_dataset("HFXM/SyntheticScoredDataPairwise_Labeled", split="train[80000:]")
# print(train_dataset)

# df = train_dataset.to_pandas()
# df = df.drop(columns=['CHOSEN', 'REJECTED', 'labels'])

# train_dataset_df = load_dataset('Anthropic/hh-rlhf')['train']
# validation_dataset_df = load_dataset('Anthropic/hh-rlhf')['test'].to_pandas()

# random.seed(0)
# train_index = random.sample(range(len(train_dataset_df)), 5000)
# train_dataset_df_subset = train_dataset_df.loc[train_index]
# df = train_dataset_df_subset.to_pandas()
# df = df.drop(columns=['CHOSEN', 'REJECTED', 'labels'])

train_dataset = load_dataset('mingye94/human_labeled_prefdata')['pku']
# validation_dataset_df = load_dataset('Anthropic/hh-rlhf')['test'].to_pandas()

# Set a random seed for reproducibility
# random.seed(0)

# # Select a subset of 5000 examples from the training dataset
# train_index = random.sample(range(len(train_dataset)), 5000)
# train_dataset_subset = train_dataset.select(train_index)

# Convert to pandas DataFrame and drop unnecessary columns
df = train_dataset.to_pandas()
# df = df.drop(columns=['CHOSEN', 'REJECTED', 'labels'])



# validation_dataset_df_subset = validation_dataset_df[:1000]

# train_dataset = Dataset.from_pandas(train_dataset_df_subset)
# validation_dataset = Dataset.from_pandas(validation_dataset_df_subset)

# print("train_dataset=\n", train_dataset)
# print("validation_dataset=\n", validation_dataset)



print("\n\n==========Apply trained RuleAdapter to generate the 5 rules.=============")

#Load model
# model_name = "/n/netscratch/lu_lab/Lab/xiaominli/LLMResearch/LLM-RLHF/RuleAdapter/train_rule_adapter/trainedRA/RuleAdapter-50K-2e-5-8epoch" # This is not best
model_name = "HFXM/RuleAdapter-final"
# model_name = "HFXM/RuleAdapter-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained('mingye94/3_rules_RA_lr1e-05_epoch3_50K')
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=100,                # 100-dimensional output
    problem_type="multi_label_classification",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.pad_token_id)


def inference(text, num_rules=5):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # Forward pass, now both inputs and model are on the same device
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(**inputs)

    # Get logits from the output
    logits = output.logits
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.float()  # Convert from bfloat16 to float32
    probabilities = probabilities.cpu().numpy()


    # Option1: Fix prediction to be 5 rules:
    top_k = num_rules
    batch_size, num_labels = probabilities.shape
    sorted_indices = np.argsort(probabilities, axis=1)  # shape (batch_size, num_labels)
    top_k_indices = sorted_indices[:, -top_k:]  # shape (batch_size, 5) in ascending order

    predictions = np.zeros_like(probabilities, dtype=int)
    for i in range(batch_size):
        predictions[i, top_k_indices[i]] = 1
    # selected_rules = np.array(rule_pool)[predictions[0] == 1]
    # print("Selected rules:", selected_rules)
    selected_rule_indices = np.where(predictions[0] == 1)[0]
    selected_rule_indices = list(map(int, selected_rule_indices)) # originally a numpy.int64 array, convert to int list
    return selected_rule_indices

        
def RuleAdapter_input_template(prompt, responseA, responseB):
        return f"Prompt:\n{prompt}\n\nResponse A:\n{responseA}\n\nResponse B:\n{responseB}"
# Batch inference
def applyRuleAdapter(start_batch_index, end_batch_index, batch_size=1000):
    data_selected_rules_folder = 'data_selected_rules'
    os.makedirs(data_selected_rules_folder, exist_ok=True)

    df['text'] = df.apply(lambda row: RuleAdapter_input_template(row['prompt'], row['chosen'], row['rejected']), axis=1)
    
    total_samples = len(df)
    total_batches = (total_samples + batch_size - 1) // batch_size
    print(f"\ntotal_batches = {total_batches}, so valid batch index range is 0-{total_batches-1}")

    if end_batch_index is None or end_batch_index==-1 or end_batch_index > total_batches: end_batch_index = total_batches - 1
    print(f"start_batch_index = {start_batch_index}, end_batch_index = {end_batch_index}")


    for batch_num in tqdm(range(start_batch_index, end_batch_index+1), desc=f"Process {end_batch_index - start_batch_index + 1} batches", total=end_batch_index - start_batch_index + 1):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_samples)
        print(f"DEBUG: for batch{batch_num}, start_idx = {start_idx}, end_idx = {end_idx}")
        save_file_path = f'{data_selected_rules_folder}/RASelectedRules_{batch_num}.jsonl'
        if os.path.exists(save_file_path):
            print(f"File {save_file_path} already exists, skip this batch")
            continue

        df_subset = df.iloc[start_idx:end_idx]
        selected_rule_indices_ls = []
        # df_subset['text'] = df_subset.apply(lambda row: RuleAdapter_input_template(row['prompt'], row['chosen'], row['rejected']), axis=1)
        # print(df['text'][0])
        for i, row in tqdm(df_subset.iterrows(), total=len(df_subset)): # here df_subset keeps the original index
            # text = df.apply(lambda row: RuleAdapter_input_template(row['prompt'], row['chosen'], row['rejected']), axis=1)
            text = row['text']
            selected_rule_indices = inference(text)
            selected_rule_indices_ls.append({'idx': i, 'RA_selected_rule_indices': selected_rule_indices})
        
        # Write each dictionary in the list to a new line in the file
        with open(save_file_path, 'w') as file:
            for item in selected_rule_indices_ls:
                file.write(json.dumps(item) + '\n')
        print(f"Saved batch {batch_num} to {save_file_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_batch_index', type=int, default=0, help="starting batch index")
    parser.add_argument('end_batch_index', type=int, default=-1, help="ending batch index (inclusive)")

    args = parser.parse_args()

    start_batch_index = args.start_batch_index
    end_batch_index = args.end_batch_index
    applyRuleAdapter(start_batch_index, end_batch_index)