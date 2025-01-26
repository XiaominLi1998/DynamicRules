import random
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments
from trl import RewardTrainer
from transformers import Trainer
import os
import argparse

print("\n\n\n\n*********************************************************************************")
print("*********************START trainRewardModel.py********************")
print("*********************************************************************************")

# Initialize the parser
parser = argparse.ArgumentParser(description="Train a model with given parameters")

# Add arguments to the parser
parser.add_argument('--num_rules', type=int, default=3, help='Num of rules')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--trial_index', type=int, default=0, help='Trial index')
parser.add_argument('--RM_train_data_path', type=str, default="data/default-RM_training_data_3rules_i15.jsonl", help='Path to save the trained RA model')
parser.add_argument('--RM_save_path', type=str, default="trainedRM", help='Path to save the trained RM model')
parser.add_argument('--use_hf_data',  type=bool, default=False, help='Flag to load data directly from Hugging Face')


# Parse the command line arguments
args = parser.parse_args()
num_rules = args.num_rules
learning_rate = args.lr
num_train_epochs = args.epochs
trial_index = args.trial_index
RM_train_data_path = args.RM_train_data_path
RM_save_path = args.RM_save_path
use_hf_data = args.use_hf_data



def format_scientific(n):
    return f"{n:.0e}".replace("e-0", "e-").replace("e+0", "e+")

#model_save_path = RM_save_path
data_path = RM_train_data_path

#print("Final RM: model_save_path = ", model_save_path)
print("data_path = ", data_path)
print("num_rules = ", num_rules)
print("learning_rate = ", learning_rate)
print("num_train_epochs = ", num_train_epochs)



print("\n\n\n==============================Load Backbone Model to trainRM===========================")

hf_token = os.getenv('HF_TOKEN')
#model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.bfloat16, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print(f"\nDone loading model {model_name}!")



print("\n\n\n==============================Load RM training data===========================")
print("Loading data from local path...")
df_trainRM = pd.read_json(data_path, lines=True)
train_dataset = Dataset.from_pandas(df_trainRM)
print("train_dataset=\n", train_dataset)



print("\n\n\n==============================Tokenize RM training data===========================")
def tokenize_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen_str, rejected_str in zip(examples['chosen'], examples['rejected']):
        chosen_tokenized = tokenizer(chosen_str, truncation=True, padding='max_length', max_length=1024)
        rejected_tokenized = tokenizer(rejected_str, truncation=True, padding='max_length', max_length=1024)
        new_examples["input_ids_chosen"].append(chosen_tokenized["input_ids"])
        new_examples["attention_mask_chosen"].append(chosen_tokenized["attention_mask"])
        new_examples["input_ids_rejected"].append(rejected_tokenized["input_ids"])
        new_examples["attention_mask_rejected"].append(rejected_tokenized["attention_mask"])
    return new_examples

train_dataset_processed = train_dataset.map(tokenize_function, batched=True)
print("\nDone tokenizing dataset!")



print("\n\n\n==============================Train RM===============================")
# Configuring the training arguments
training_args = TrainingArguments(
    output_dir="trainedRM/Temp-RM",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    warmup_steps=10,
    report_to='none',
    evaluation_strategy="no",
    save_strategy="no",
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=False,
    bf16=True
)

training_args.max_length = 512

# Loading the RewardTrainer from TRL
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset_processed,
    # eval_dataset=validation_dataset_processed
)
trainer.train()
print("\nDone RewardModel training!")




print("\n\n\n==============================Save trainedRM Model locally===============================")

model_save_path = RM_save_path
print("Final RM: model_save_path = ", model_save_path)
# Save the trained model
model.save_pretrained(model_save_path)
print(f"Saved trained RM lcoally to: {model_save_path}")
tokenizer.save_pretrained(model_save_path)
print(f"Saved tokenizer locally to: {model_save_path}")


print("\n\n\n\n*********************************************************************************")
print("*********************END trainRewardModel.py********************")
print("*********************************************************************************")
