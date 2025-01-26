
import torch
import numpy as np
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from templates import rule_rating_template
from rules.rule_pool import rule_pool
from tqdm import tqdm
from datasets import load_dataset, Dataset
from templates import ShareGPT_rating_template2, rule_rating_logits_template
import pandas as pd
from tqdm import tqdm
import os
import sys
import time
import argparse




print("\n\n\n==============================Load data==============================")
# Load the dataset
dataset = load_dataset('data_with_responses') # dataset where each sample contains 1 question and 6 responses 
print("Loaded dataset: \n", dataset)


df = dataset['train'].to_pandas()
score_response_name_map = {'alpaca_7B_scores': 'PKU-Alignment/alpaca-7b-reproduced',
'llama2_7B_scores' : 'meta-llama/Llama-2-7b-chat-hf',
'mistral_7B_scores' : 'mistralai/Mistral-7B-Instruct-v0.3',
'gpt_4omini_scores' : 'gpt-4o-mini',
'mixtral_8x7B_scores' : 'mistralai/Mixtral-8x7B-Instruct-v0.1',
'llama3_70B_scores' : 'meta-llama/Meta-Llama-3-70B-Instruct'
}

for key in score_response_name_map:
    df[key] = None
print("df shape", df.shape)



print("\n\n\n==============================Load model==============================")
# model_name = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"
# model_name = "failspy/llama-3-70B-Instruct-abliterated"
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval() 




print("\n\n\n==============================Start rating==============================")

yes_token = ' Yes'
no_token = ' No'
yes_ids = tokenizer(yes_token, add_special_tokens=False)["input_ids"]
no_ids  = tokenizer(no_token,  add_special_tokens=False)["input_ids"]



def yes_probability(sentence):
    # Tokenize
    tokens = tokenizer(sentence, return_tensors="pt")

    # Get logits for each position
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits[0]  # Remove batch dimension [seq_len, vocab_size]

    # Get probability distribution at the position '-1'
    probs = F.softmax(logits[-1], dim=-1)

    # Get probability of actual token
    yes_token_prob = probs[yes_ids].item()
    no_token_prob = probs[no_ids].item()

    return yes_token_prob - no_token_prob

def rate_100rules(question, response):
    yes_prob_ls = []
    for rule in tqdm(rule_pool):
        rule = rule.replace('Accept the response that ', '')
        user_input = rule_rating_logits_template(rule, question, response)
        yes_prob = yes_probability(user_input)
        yes_prob_ls.append(yes_prob)
        sys.stdout.flush()
    return yes_prob_ls


def get_logits_scores(start_batch_index, end_batch_index):
    batch_size = 1000
    total_prompts = len(df)
    total_batches = (total_prompts + batch_size - 1) // batch_size
    print(f"\ntotal_batches = {total_batches}, so valid batch index range is 0-{total_batches-1}")

    if end_batch_index is None or end_batch_index==-1 or end_batch_index > total_batches: end_batch_index = total_batches - 1
    print("start_batch_index = ", start_batch_index)
    print("end_batch_index = ", end_batch_index)


    for batch_num in tqdm(range(start_batch_index, end_batch_index+1), desc=f"Process {end_batch_index - start_batch_index + 1} batches", total=end_batch_index - start_batch_index + 1):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_prompts)
        print(f"DEBUG: for batch{batch_num}, start_idx = {start_idx}, end_idx = {end_idx}")
        df_subset = df.iloc[start_idx:end_idx]

        for index, row in tqdm(df_subset.iterrows(), desc=f"Batch{batch_num}", total=batch_size):
            print(f"\n\n-----------------------Data sample {index}-----------------------")
            prompt = row['prompt']

            for score_coln, response_coln in score_response_name_map.items():
                response = row[response_coln]
                yes_prob_ls = rate_100rules(prompt, response)
                df_subset.at[index, score_coln] = yes_prob_ls
        
        save_file_path = f'rating_scores/Llama70B_yes_probs_batch_{start_batch_index}_{end_batch_index}.jsonl'
        df_subset.to_json(save_file_path, orient='records', lines=True)
        print(f"Saved batch {batch_num} to {save_file_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_batch_index', type=int, default=0, help="starting batch index")
    parser.add_argument('end_batch_index', type=int, default=-1, help="ending batch index (inclusive)")

    args = parser.parse_args()

    start_batch_index = args.start_batch_index
    end_batch_index = args.end_batch_index
    get_logits_scores(start_batch_index, end_batch_index)