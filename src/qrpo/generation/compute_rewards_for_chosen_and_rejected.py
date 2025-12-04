import logging
import math
import os
from pathlib import Path

import datasets
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


def compute_chosen_and_rejected_rewards_batch(reward_model, batch, tokenizer):
    """
    Takes in one batch (from the filtered data), prepares inputs,
    then does a single forward pass to get chosen/rejected scores.
    """
    inputs = sum([[row["chosen"]] + [row["rejected"]] for row in batch], start=[])

    tokenized_inputs = tokenizer.apply_chat_template(
        inputs,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        output = reward_model(
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
        )

    # shape = (2 * batch_size) -> (batch_size, 2)
    rewards = output.score.cpu().float().reshape(-1, 2)
    chosen_rewards = rewards[:, 0].tolist()
    rejected_rewards = rewards[:, 1].tolist()

    return chosen_rewards, rejected_rewards


@hydra.main(
    config_path="../configs",
    config_name="compute-rewards-for-chosen-and-rejected",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    utils.seeding.seed_everything(config)

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

    data = datasets.load_from_disk(config.dataset_args.dataset_path)[config.split]

    # To filter out rows that are too long
    def add_chat_num_tokens(row):
        chosen_tokens = tokenizer.apply_chat_template(row["chosen"], tokenize=True)
        rejected_tokens = tokenizer.apply_chat_template(row["rejected"], tokenize=True)
        chosen_text = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        rejected_text = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return {
            "chosen_reward_tokens": chosen_tokens,
            "chosen_reward_tokens_len": len(chosen_tokens),
            "chosen_reward_text": chosen_text,
            "rejected_reward_tokens": rejected_tokens,
            "rejected_reward_tokens_len": len(rejected_tokens),
            "rejected_reward_text": rejected_text,
            "max_chosen_rejected_reward_tokens_len": max(
                len(chosen_tokens), len(rejected_tokens)
            ),
        }

    data = data.map(add_chat_num_tokens, num_proc=64)
    # Filter out the row with chat_num_tokens > training_args.max_seq_length
    data = data.filter(
        lambda max_chosen_rejected_reward_tokens_len: max_chosen_rejected_reward_tokens_len
        <= config.max_seq_length,
        input_columns=["max_chosen_rejected_reward_tokens_len"],
    )

    # Record the size of the dataset after filtering
    logger.info(f"Filtered dataset size: {len(data)}")

    if config.dataset_args.debug_oom:
        data = data.sort("max_chosen_rejected_reward_tokens_len", reverse=True)

    subpartition_size = math.ceil(len(data) / 4)
    subpartition_start_idx = config.subpartition_number * subpartition_size
    subpartition_end_idx = (config.subpartition_number + 1) * subpartition_size
    subpartition_end_idx = min(subpartition_end_idx, len(data))

    subpartition_data = data.select(range(subpartition_start_idx, subpartition_end_idx))

    # Handle resuming.
    resuming_dir = Path.cwd()
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

    local_start_idx = already_processed_samples
    if local_start_idx > 0:
        logger.info(
            f"Resuming from checkpoint-{local_start_idx}. Processing from sample {local_start_idx}."
        )

    pbar = tqdm(
        total=len(subpartition_data),
        desc="Computing rewards for chosen and rejected completions",
    )
    pbar.update(local_start_idx)

    while local_start_idx < len(subpartition_data):
        local_end_idx = min(
            local_start_idx + config.save_interval, len(subpartition_data)
        )
        current_slice = (local_start_idx, local_end_idx)
        current_slice_data = subpartition_data.select(range(*current_slice))

        # Compute chosen and rejected rewards for the current slice
        chosen_rewards = []  # final len = len(current_slice_data)
        rejected_rewards = []  # final len = len(current_slice_data)
        num_batches = math.ceil(len(current_slice_data) / config.batch_size)
        for batch_idx in trange(num_batches):
            batch = current_slice_data.select(
                range(
                    config.batch_size * batch_idx,
                    min(config.batch_size * (batch_idx + 1), len(current_slice_data)),
                )
            )
            (
                chosen_rewards_batch,
                rejected_rewards_batch,
            ) = compute_chosen_and_rejected_rewards_batch(
                reward_model, batch, tokenizer
            )
            chosen_rewards += chosen_rewards_batch
            rejected_rewards += rejected_rewards_batch

        # Save the current slice.
        current_slice_data = current_slice_data.add_column(
            "chosen_rewards", chosen_rewards
        )
        current_slice_data = current_slice_data.add_column(
            "rejected_rewards", rejected_rewards
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
