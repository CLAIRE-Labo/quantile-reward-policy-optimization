import json
import logging
import random
import time
from pathlib import Path

import datasets
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from qrpo import utils
from qrpo.generation.prepare_code_dataset import pre_code
from sandbox.client import SandboxClient

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


# Function to compute rewards for a single row
def compute_rewards_for_rows(reward_model, tokenizer, batch, config):
    batch_prompts = [sample[0] for sample in batch["ref_completions"]]
    batch_entrypoints = batch["entry_point"]
    batch_ref_model_completions = [
        json.loads(sample[1]["content"]) for sample in batch["ref_completions"]
    ]
    batch_tests = batch["test"]

    # Extract the tokenized sequences without padding
    batch_ref_completions_reward_tokens = []
    batch_ref_completions_reward_tokens_len = []
    batch_ref_completions_reward_texts = []

    for prompt, ref_model_completions in zip(
        batch_prompts, batch_ref_model_completions
    ):
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
        )

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
        batch_ref_completions_reward_tokens.append(ref_completions_reward_tokens)
        batch_ref_completions_reward_tokens_len.append(
            ref_completions_reward_tokens_len
        )
        batch_ref_completions_reward_texts.append(ref_completions_reward_texts)

    batch_ref_rewards = reward_model.execute_batch_code(
        pre_code,
        batch_ref_model_completions,  # [B, Nref, str]
        batch_entrypoints,  # [B, str]
        batch_tests,  # [B, str]
        config.reward_model_args,
    )
    return (
        batch_ref_rewards,
        batch_ref_completions_reward_tokens,
        batch_ref_completions_reward_tokens_len,
        batch_ref_completions_reward_texts,
    )


def compute_rewards_batch(reward_model, tokenizer, data, config):
    ref_rewards = []
    ref_completions_reward_tokens = []
    ref_completions_reward_tokens_len = []
    ref_completions_reward_texts = []

    num_prompts_per_batch = config.batch_size // config.dataset_num_ref_rewards
    for i in trange(0, len(data), num_prompts_per_batch):
        batch = data.select(range(i, min(i + num_prompts_per_batch, len(data))))
        (
            batch_rewards,
            batch_completions_reward_tokens,
            batch_completions_reward_tokens_len,
            batch_completions_reward_texts,
        ) = compute_rewards_for_rows(reward_model, tokenizer, batch, config)
        ref_rewards.extend(batch_rewards)
        ref_completions_reward_tokens.extend(batch_completions_reward_tokens)
        ref_completions_reward_tokens_len.extend(batch_completions_reward_tokens_len)
        ref_completions_reward_texts.extend(batch_completions_reward_texts)
    return (
        ref_rewards,
        ref_completions_reward_tokens,
        ref_completions_reward_tokens_len,
        ref_completions_reward_texts,
    )


@hydra.main(
    config_path="../configs",
    config_name="compute-rewards-for-solutions",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    random.seed(config.seed)

    # Create the reward model (sandbox)
    base_url = f"http://localhost:{config.reward_model_args.port}"
    reward_model = SandboxClient(base_url)
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_args.model_name_or_path, use_fast=True
    )

    # Test the connection to the sandbox server
    while True:
        time.sleep(1)
        try:
            reward_model.test_connection()
            logger.info("Sandbox server is ready!")
            break
        except Exception as e:
            logger.warning(f"Sandbox server not ready yet. Retrying... {e}")

    # Load dataset
    # End index is exclusive
    subpartition_start_idx = config.partition_start_idx
    subpartition_end_idx = config.partition_end_idx

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
        current_slice_data = subpartition_data.select(range(*current_slice))
        (
            ref_rewards,
            ref_completions_reward_tokens,
            ref_completions_reward_tokens_len,
            ref_completions_reward_texts,
        ) = compute_rewards_batch(reward_model, tokenizer, current_slice_data, config)
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
