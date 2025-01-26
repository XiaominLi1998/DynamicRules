import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
import random
from transformers import TrainingArguments, Trainer,  DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
from tqdm import tqdm
from rules.rule_pool import rule_pool


print("\n\n\n\n*********************************************************************************")
print("*********************START trainRuleAdapter.py********************")
print("*********************************************************************************")

# Initialize the parser
parser = argparse.ArgumentParser(description="Train a model with given parameters")

# Add arguments to the parser
parser.add_argument('--num_rules', type=int, default=5, help='Num of rules')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--trial_index', type=int, default=0, help='Trial index')
parser.add_argument('--RA_train_data_path', type=str, default="RA_training_data", help='RA training data save path')
parser.add_argument('--pairwise_dataset_name', type=str, default="pairwise_dataset", help='Huggingface pairiwse data with rating scores')
parser.add_argument('--RA_save_path', type=str, default="trainedRA/RA", help='Path to save the trained RA model')
parser.add_argument('--RandomRules', action='store_true', help='Randomly select rules or not')
parser.add_argument('--Debug', action='store_true', help='Enable debug mode')

# Parse the command line arguments
args = parser.parse_args()
num_rules = args.num_rules
learning_rate = args.lr
num_train_epochs = args.epochs
trial_index = args.trial_index
RA_train_data_path = args.RA_train_data_path
pairwise_dataset_name = args.pairwise_dataset_name
RA_save_path = args.RA_save_path
RandomRules = args.RandomRules
Debug = args.Debug
print("num_rules = ", num_rules)
print("trial_index = ", trial_index)
print("RA_train_data_path = ", RA_train_data_path)
print("RA_save_path = ", RA_save_path)
print("RandomRules = ", RandomRules)
print("Debug = ", Debug)

model_save_path = RA_save_path


print(f"\n\n=======================Prepare RA train data with NumRules{num_rules}========================")

dataset_dict = load_dataset(pairwise_dataset_name)
train_dataset = dataset_dict['train']
val_dataset = dataset_dict['validation']
test_dataset = dataset_dict['test']
if Debug: 
    train_dataset = train_dataset.select(range(200))  # DEBUG
    test_dataset = test_dataset.select(range(100))  # DEBUG


def generate_new_labels(dataset, num_rules):
    df = dataset.to_pandas()
    for i, row in tqdm(df.iterrows(), total=len(df)):

        # Use relevance score and discrepancy: score(CHOSEN) - score(REJECT) to select out final rules    
        relevance_scores = np.array(row['relevance'])
        scoresA  = np.array(row['scoresA'])
        scoresB  = np.array(row['scoresB'])

        if RandomRules:
            # Select random rules
            if num_rules == 100: #All 100 rules 
                random_indices = range(100)
            else: #Random 5 rules
                random_indices = random.sample(range(100), num_rules)
            top_indices = np.array(random_indices)
        else:
            # Use relevance score and discrepancy: |scoreA - scoreB| to select out final rules    
            discrepancy = scoresA - scoresB
            abs_discrepancy = np.abs(discrepancy)
            sum_terms = relevance_scores + 1/2 * abs_discrepancy
            # Select top rules
            top_indices = sum_terms.argsort()[::-1][:num_rules]


        final_rules = [rule_pool[idx] for idx in top_indices]
        df.at[i, 'rule_indices'] = list(top_indices)
        df.at[i, 'final_rules'] = final_rules

        # Create labels to trainRA
        rewardA = np.sum(scoresA[top_indices])
        rewardB = np.sum(scoresB[top_indices])
        if rewardA > rewardB:
            df.at[i, 'CHOSEN'] = row['responseA_model']
            df.at[i, 'REJECTED']  = row['responseB_model']
        else:
            df.at[i, 'CHOSEN'] = row['responseB_model']
            df.at[i, 'REJECTED']  = row['responseA_model']

        labels = np.zeros(100)
        labels[top_indices] = 1
        df.at[i, 'labels'] = labels

    return Dataset.from_pandas(df)


train_dataset = generate_new_labels(train_dataset, num_rules)
val_dataset = generate_new_labels(val_dataset, num_rules)
test_dataset = generate_new_labels(test_dataset, num_rules)

dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
dataset_dict.save_to_disk(RA_train_data_path)

# Print out sizes of the datasets to verify
print("dataset_dict = ", dataset_dict)



print("\n\n=======================Load Base Model to trainRA=======================")
# model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-3.2-3B"
train_batch_size = 8

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=100,                # 100-dimensional output
    problem_type="multi_label_classification",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.pad_token_id)



print("\n\n\n======================= Tokenize RA training data=======================")
def tokenize_function(example):
    # Standard tokenization
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,  # adjust as needed
        padding="max_length",
    )
    # Keep the labels as-is (already float vectors)
    tokenized["labels"] = example["labels"]
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=False)
val_dataset  = val_dataset.map(tokenize_function, batched=False)
test_dataset  = test_dataset.map(tokenize_function, batched=False)

# Set 'input_ids' and 'attention_mask' as PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])





print("\n\n\n======================= Training RA=======================")
def compute_metrics(eval_pred):
    """
    Computes metrics for multi-label classification using a top-k strategy.
    
    Args:
    eval_pred: tuple (predictions, labels)
        predictions: logits of shape (batch_size, num_labels)
        label_ids: ground-truth labels of shape (batch_size, num_labels)
    
    Returns:
    dict: Dictionary containing accuracy, F1, precision, and recall.
    """
    predictions, labels = eval_pred
    batch_size, num_labels = predictions.shape
    
    # Convert logits to probabilities via sigmoid
    probs = 1 / (1 + np.exp(-predictions))  # Same as torch.sigmoid, but in NumPy
    
    # For each sample, pick the indices of the top-5 probabilities
    top_k = num_rules
    partitioned_indices = np.argpartition(probs, -top_k, axis=1)
    top_k_indices = partitioned_indices[:, -top_k:]
    
    # Build a binary predictions array with exactly 5 positions set to 1
    preds = np.zeros_like(probs, dtype=int)
    for i in range(batch_size):
        preds[i, top_k_indices[i]] = 1
    
    # Flatten the predictions and labels for computing the metrics globally
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    # Compute the metrics
    accuracy = accuracy_score(labels_flat, preds_flat)
    f1_micro = f1_score(labels_flat, preds_flat, average='micro')
    f1_macro = f1_score(labels_flat, preds_flat, average='macro')
    precision_micro = precision_score(labels_flat, preds_flat, average='micro')
    precision_macro = precision_score(labels_flat, preds_flat, average='macro')
    recall_micro = recall_score(labels_flat, preds_flat, average='micro')
    recall_macro = recall_score(labels_flat, preds_flat, average='macro')
    
    return {
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro
    }


# per_device_train_batch_size = 8
# per_device_eval_batch_size = 8
per_device_train_batch_size = train_batch_size
per_device_eval_batch_size = train_batch_size
logging_steps=10 * (8 // per_device_train_batch_size)
eval_steps=100 * (8 // per_device_train_batch_size)
save_strategy="steps"
save_steps = int(40000 / per_device_train_batch_size)

# #DEBUG:
# num_train_epochs = 1
# per_device_train_batch_size = 1
# per_device_eval_batch_size = 1
# logging_steps=1
# eval_steps=10
# save_strategy="no"

training_args = TrainingArguments(
    output_dir="./temp/trainRuleAdapter-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,            # Example number of epochs
    per_device_train_batch_size=per_device_train_batch_size,  # Adjust based on your GPU capabilities
    per_device_eval_batch_size=per_device_eval_batch_size,
    logging_steps=logging_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    logging_dir='./temp/trainRuleAdapter-logs',          # Where to store log files
    learning_rate=learning_rate,
    weight_decay=0.01,
    report_to=None,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)


# Assuming the tokenizer is correctly initialized
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Use the appropriate data collator
    compute_metrics=compute_metrics  # Define this if needed for evaluation
)


trainer.train()

# Save the model and the tokenizer at the end of training
print("\n\n\n======================= Saving trained RA model locally=======================")
 
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Saved trained RA locally to: {model_save_path}!")


print("\n\n\n\n*********************************************************************************")
print("*********************END trainRuleAdapter.py********************")
print("*********************************************************************************")
