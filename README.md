# DynamicRules
Reward model training based on dynamic rules

To label existing pairewise preference dataset:
1. Select most critical rules by applying the rule adapter to your preference data:
   python applyRuleAdapter.py

2. Give rating according to the selected rules and obtain preference label:
   python rating.py

3. Train reward model based on the labeled preference dataset:
   python trainRewardModel.py

To train your own rule adapter with existing preference dataset:
1. Give rating according to 100 rules in the rule pool:
   python rating.py 

2. Train rule adapter:
   run trainRuleAdpter.py


Our reward model RAMO: https://huggingface.co/HFXM/RAMO-Llama3.1-8B

Our RuleAdapter: https://huggingface.co/HFXM/RAMO-RuleAdapter
