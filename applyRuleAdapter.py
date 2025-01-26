from datasets import load_dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import torch
#from rules.rule_pool import rule_pool
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

print("\n\n\n\n*********************************************************************************")
print("*********************START applyRuleAdapter.py********************")
print("*********************************************************************************")



# Initialize the parser
parser = argparse.ArgumentParser(description="Train a model with given parameters")
parser.add_argument('--num_rules', type=int, default=3, help='Num of rules')
parser.add_argument('--trial_index', type=int, default=0, help='Trial index')
parser.add_argument('--RA_save_path', type=str, default="RuleAdapter", help='Path to save the trained RA model')
parser.add_argument('--RM_train_data_path', type=str, default="data/default-RM_training_data.jsonl", help='Path to save the trained RA model')
parser.add_argument('--Debug', action='store_true', help='Enable debug mode')

args = parser.parse_args()
num_rules = args.num_rules
trial_index = args.trial_index
RA_save_path = args.RA_save_path
RM_train_data_path = args.RM_train_data_path
Debug = args.Debug

model_path =  RA_save_path
print("load trainedRA from: model_path = ", model_path)
print("RM_train_data_path = ", RM_train_data_path)
print("Debug = ", Debug)




print("\n\n==========Load dataset to applyRA+trainRM.=============")
i=0
ShareGPT_data_size = 1000
start = i * ShareGPT_data_size
end = (i + 1) * ShareGPT_data_size
if Debug: ShareGPT_data_size = 200
train_dataset = load_dataset("HFXM/RewardModel_training_data", split=f"ShareGPT[{start}:{end}]")
df = train_dataset.to_pandas()
df = df.drop(columns=['rule_indices', 'final_rules'])
print("df.shape = ", df.shape)
print("df.columnes = ", df.columns)



print(f"\n\n==========Load trained RA.=============")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
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


    # Fix prediction to be 5 rules:
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

        

print(f"\n\n==========Apply RA to generate the {num_rules} rules and preference labels.=============")
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    selected_rule_indices = inference(text)
    df.at[i, 'RA_selected_rule_indices'] = selected_rule_indices
    
    # Use relevance score and discrepancy: score(CHOSEN) - score(REJECT) to select out final rules    
    scoresA  = np.array(row['scoresA'])
    scoresB  = np.array(row['scoresB'])
    rewardA = np.sum(scoresA[selected_rule_indices])
    rewardB = np.sum(scoresB[selected_rule_indices])
    qaA = f"Human: {row['prompt']}\n\nAssistant: {row['responseA']}"
    qaB = f"Human: {row['prompt']}\n\nAssistant: {row['responseB']}"
    if rewardA > rewardB:
        df.at[i, 'chosen'] = qaA
        df.at[i, 'rejected']  = qaB
    else:
        df.at[i, 'chosen'] = qaB
        df.at[i, 'rejected']  = qaA


df.to_json(RM_train_data_path, orient='records', lines=True)
print(f"Saved the training data with {num_rules} rules to {RM_train_data_path}.")



print("\n\n\n\n*********************************************************************************")
print("*********************END applyRuleAdapter.py********************")
print("*********************************************************************************")
