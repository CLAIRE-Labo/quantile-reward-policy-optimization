# Data

## Base datasets

A reference of the commands used to obtain the initial datasets and models.

```bash
# Datasets

# magpieair
python -c "import datasets; datasets.load_dataset('Magpie-Align/Magpie-Air-DPO-100K-v0.1').save_to_disk('data/shared/datasets/magpieair')"
# ultrafeedback
python -c "import datasets; datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized').save_to_disk('data/shared/datasets/ultrafeedback')"
# Alpaca
python -c "import datasets; datasets.load_dataset('tatsu-lab/alpaca_eval').save_to_disk('data/shared/datasets/alpaca')"
# LeetCodeDataset
python -c "import datasets; datasets.load_dataset('newfacade/LeetCodeDataset', revision='a9eca795817a0a21132070e2dc2e87445da4f089').save_to_disk('data/shared/datasets/leetcode')"

# models
# RLHFlow/ArmoRM-Llama3-8B-v0.1
huggingface-cli download RLHFlow/ArmoRM-Llama3-8B-v0.1 --local-dir data/shared/models/armorm
# allenai/Llama-3.1-Tulu-3-8B-SFT
huggingface-cli download allenai/Llama-3.1-Tulu-3-8B-SFT --local-dir data/shared/models/llama
# mistralai/Mistral-7B-Instruct-v0.2
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir data/shared/models/mistral
```

## Datasets with references completions and rewards

We will release the datasets with references completions and reference rewards for all dataset-model combinations soon.
