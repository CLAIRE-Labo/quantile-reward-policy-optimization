import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


# Function to compute rewards for a single row
def compute_rewards_for_row(reward_model, sample, tokenizer, config):
    prompt = sample[0]
    ref_model_completions = json.loads(sample[1]["content"])
    ref_inputs = [
        [prompt] + [{"role": "assistant", "content": completion}]
        for completion in ref_model_completions
    ]

    tokenized_ref_inputs = tokenizer.apply_chat_template(
        ref_inputs,
        return_tensors="pt",
        padding=True,
        return_dict=True,
        truncation=True,
        max_length=config.max_seq_length,
    )

    # Extract the tokenized sequences without padding
    ref_completions_reward_tokens = []
    ref_completions_reward_tokens_len = []
    ref_completions_reward_texts = []

    for i in range(len(ref_inputs)):
        # Get the attention mask for this input
        attention_mask = tokenized_ref_inputs["attention_mask"][i]
        # Get the number of tokens (excluding padding)
        tokens_len = torch.sum(attention_mask).item()
        # Get the actual tokens (excluding padding)
        tokens = tokenized_ref_inputs["input_ids"][i][:tokens_len].tolist()
        # This is the text used by the reward model to get the reward.
        text = tokenizer.decode(tokens, skip_special_tokens=False)

        ref_completions_reward_tokens.append(tokens)
        ref_completions_reward_tokens_len.append(tokens_len)
        ref_completions_reward_texts.append(text)

    num_iters = math.ceil(len(ref_model_completions) / config.batch_size)

    ref_rewards = []
    for i in range(num_iters):
        with torch.no_grad():
            output = reward_model(
                input_ids=tokenized_ref_inputs["input_ids"][
                    i * config.batch_size : (i + 1) * config.batch_size
                ].to("cuda"),
                attention_mask=tokenized_ref_inputs["attention_mask"][
                    i * config.batch_size : (i + 1) * config.batch_size
                ].to("cuda"),
            )
            ref_rewards.append(output.score.cpu().float())

    ref_rewards = torch.cat(ref_rewards, dim=0).numpy()
    return (
        ref_rewards.tolist(),
        ref_completions_reward_tokens,
        ref_completions_reward_tokens_len,
        ref_completions_reward_texts,
    )


def compute_rewards_batch(batch, reward_model, tokenizer, config):
    ref_rewards = []
    ref_completions_reward_tokens = []
    ref_completions_reward_tokens_len = []
    ref_completions_reward_texts = []
    for sample in tqdm(batch, desc="Processing batch"):
        (
            rewards,
            completions_reward_tokens,
            completions_reward_tokens_len,
            completions_reward_texts,
        ) = compute_rewards_for_row(reward_model, sample, tokenizer, config)
        ref_rewards.append(rewards)
        ref_completions_reward_tokens.append(completions_reward_tokens)
        ref_completions_reward_tokens_len.append(completions_reward_tokens_len)
        ref_completions_reward_texts.append(completions_reward_texts)
    return (
        ref_rewards,
        ref_completions_reward_tokens,
        ref_completions_reward_tokens_len,
        ref_completions_reward_texts,
    )


def compute_subpartition_start_end_indices(
    partition_start_idx, partition_end_idx, subpartition_number
):
    subpartition_size = math.ceil((partition_end_idx - partition_start_idx) / 4)
    start_idx = partition_start_idx + subpartition_number * subpartition_size
    end_idx = partition_start_idx + (subpartition_number + 1) * subpartition_size
    end_idx = min(end_idx, partition_end_idx)

    return start_idx, end_idx


@hydra.main(
    config_path="../configs",
    config_name="compute-rewards-for-completions",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    random.seed(config.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.subpartition_number)
    logger.info(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Load reward model + tokenizer
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_args.model_name_or_path,
        device_map="cuda",
        trust_remote_code=config.reward_model_args.trust_remote_code,
        torch_dtype=config.reward_model_args.torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_args.model_name_or_path, use_fast=True
    )

    # End index is exclusive
    (
        subpartition_start_idx,
        subpartition_end_idx,
    ) = compute_subpartition_start_end_indices(
        config.partition_start_idx, config.partition_end_idx, config.subpartition_number
    )

    if subpartition_start_idx >= subpartition_end_idx:
        logger.info("Subpartition is empty. Exiting.")
        return

    subpartition_data = datasets.load_from_disk(config.dataset_path)[
        config.split
    ].select(range(subpartition_start_idx, subpartition_end_idx))

    # Handle resuming.
    resuming_dir = Path.cwd()
    # Checkpoints are saved as `checkpoint-{last-relative-index-processed-in-the-subpartition}`.
    already_processed_samples = max(
        (
            int(item.name.split("-")[-1])
            for item in resuming_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint-")
        ),
        default=0,
    )
    if already_processed_samples == len(subpartition_data):
        logger.info(
            "All samples in the subpartition have already been processed. Exiting."
        )
        return

    local_start_idx = already_processed_samples  # 64, 128, ...
    if local_start_idx > 0:
        logger.info(
            f"Resuming from checkpoint-{local_start_idx}. Processing from sample {local_start_idx}."
        )

    pbar = tqdm(
        total=len(subpartition_data), desc="Computing rewards for ref completions"
    )
    pbar.update(local_start_idx)
    while local_start_idx < len(subpartition_data):
        current_slice = (
            local_start_idx,
            min(local_start_idx + config.save_interval, len(subpartition_data)),
        )
        current_slice_data = subpartition_data.select(range(*current_slice))[
            "ref_completions"
        ]
        (
            ref_rewards,
            ref_completions_reward_tokens,
            ref_completions_reward_tokens_len,
            ref_completions_reward_texts,
        ) = compute_rewards_batch(current_slice_data, reward_model, tokenizer, config)
        local_end_idx = local_start_idx + len(current_slice_data)
        current_slice_data = subpartition_data.select(range(*current_slice))
        current_slice_data = current_slice_data.add_column("ref_rewards", ref_rewards)
        current_slice_data = current_slice_data.add_column(
            "ref_completions_reward_tokens", ref_completions_reward_tokens
        )
        current_slice_data = current_slice_data.add_column(
            "ref_completions_reward_tokens_len", ref_completions_reward_tokens_len
        )
        current_slice_data = current_slice_data.add_column(
            "ref_completions_reward_texts", ref_completions_reward_texts
        )

        save_path = resuming_dir / f"checkpoint-{local_end_idx}"
        current_slice_data.save_to_disk(save_path)
        logger.info(f"Saved checkpoint-{local_end_idx} successfully!")

        # Mark progress
        pbar.update(len(current_slice_data))
        local_start_idx = local_end_idx

    logger.info("Rewards computed successfully!")


if __name__ == "__main__":
    main()
