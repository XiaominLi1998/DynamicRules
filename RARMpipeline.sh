#!/bin/bash

num_rules=5
trial_index=0

# Define paths:
task_folder="."

RA_train_data_path="${task_folder}/data_temp/RuleAdapter_training_data_${num_rules}Rules"
RA_save_path="${task_folder}/trainedRA/Trial${trial_index}-${num_rules}Rules-RA-Llama3.2-3B-63K"
RM_train_data_path="${task_folder}/data_temp/Trial${trial_index}-RewardModel_training_data_${num_rules}Rules.jsonl"
RM_name="Trial${trial_index}-${num_rules}Rules-SkyworkRM-ShareGPT1000" # Needed later for reward bench
RM_save_path="${task_folder}/trainedRM/${RM_name}"


# trainRA + applyRA + trainRM:
python trainRuleAdapter.py \
  --num_rules "${num_rules}" \
  --RA_train_data_path "${RA_train_data_path}" \
  --RA_save_path "${RA_save_path}"&& \
python applyRuleAdapter.py \
  --num_rules "${num_rules}" \
  --RM_train_data_path "${RM_train_data_path}" \
  --RA_save_path "${RA_save_path}" && \
python trainRewardModel.py \
  --num_rules "${num_rules}" \
  --RM_train_data_path "${RM_train_data_path}" \
  --RM_save_path "${RM_save_path}"


# Benchmark trained RM:
mkdir -p "${task_folder}/reward_bench_results"
output_path="${task_folder}/reward_bench_results/${RM_name}.json"
python reward-bench/scripts/run_rm.py \
  --model="${RM_save_path}" \
  --chat_template=Ziya \
  --batch_size=8 \
  --output_path="${output_path}"
