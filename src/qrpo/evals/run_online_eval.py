import json
import logging
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import sleep

import datasets
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import AutoTokenizer

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


def generate_chunk_worker(
    checkpoint_path, prompts_chunk, torch_dtype, sampling_params_dict, gpu_id
):
    """
    Worker function for vLLM generation.
    Assigns a GPU via CUDA_VISIBLE_DEVICES, instantiates the LLM, and returns generated responses.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import (  # local import for subprocess pickling compatibility
        LLM,
        SamplingParams,
    )

    sampling_params = SamplingParams(**sampling_params_dict)
    llm = LLM(checkpoint_path, dtype=torch_dtype, gpu_memory_utilization=0.8)
    outputs = llm.generate(prompts_chunk, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses


def compute_rewards_worker(chunk, config, gpu_id):
    """
    Worker function to compute rewards for a chunk of completions.
    Assigns a GPU via CUDA_VISIBLE_DEVICES, loads the reward model and tokenizer,
    processes the chunk in mini-batches, and returns:
      - a list of computed rewards,
      - a list of reward token lists,
      - a list of token lengths.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load the reward model and tokenizer.
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_args.model_name_or_path,
        trust_remote_code=config.reward_model_args.trust_remote_code,
        torch_dtype=config.reward_model_args.torch_dtype,
    )
    # With CUDA_VISIBLE_DEVICES set, we can use "cuda:0"
    model.to("cuda:0")
    reward_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_args.model_name_or_path, use_fast=True
    )

    all_rewards = []
    all_reward_tokens = []
    all_reward_tokens_len = []
    all_reward_texts = []
    batch_size = config.compute_reward_config.batch_size
    num_samples = len(chunk)
    for i in tqdm(range(0, num_samples, batch_size)):
        # Select a mini-batch from the chunk.
        batch = chunk.select(range(i, min(i + batch_size, num_samples)))
        inputs = batch["prompt_and_completion"]

        tokenized_inputs = reward_tokenizer.apply_chat_template(
            inputs,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            truncation=True,
            max_length=config.compute_reward_config.max_seq_length,
        )

        # Record token information.
        for j in range(len(tokenized_inputs["attention_mask"])):
            attention_mask = tokenized_inputs["attention_mask"][j]
            tokens_len = int(torch.sum(attention_mask).item())
            tokens = tokenized_inputs["input_ids"][j][:tokens_len].tolist()
            all_reward_tokens.append(tokens)
            all_reward_tokens_len.append(tokens_len)
            all_reward_texts.append(
                reward_tokenizer.decode(tokens, skip_special_tokens=False)
            )

        # Compute rewards using the reward model.
        with torch.no_grad():
            output = model(
                input_ids=tokenized_inputs["input_ids"].to("cuda:0"),
                attention_mask=tokenized_inputs["attention_mask"].to("cuda:0"),
            )
        batch_rewards = output.score.cpu().tolist()
        all_rewards.extend(batch_rewards)
    return all_rewards, all_reward_tokens, all_reward_tokens_len, all_reward_texts


def compute_kl_divergence_worker(
    chunk, config, sampling_model_path, target_model_path, chat_key, gpu_id
):
    """
    Worker function to compute KL divergence between checkpoint model and base model.
    Assigns a GPU via CUDA_VISIBLE_DEVICES, loads both models and tokenizer,
    processes the chunk in mini-batches, and returns a list of KL divergence values.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sampling_model_path, use_fast=True)

    model_kwargs = dict(
        trust_remote_code=config.model_args.trust_remote_code,
        attn_implementation=config.model_args.attn_implementation,
        torch_dtype=config.model_args.torch_dtype,
    )
    sampling_model = AutoModelForCausalLM.from_pretrained(
        sampling_model_path,
        **model_kwargs,
    )
    sampling_model.to("cuda:0")
    sampling_model.eval()

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        **model_kwargs,
    )
    target_model.to("cuda:0")
    target_model.eval()

    sampling_model_prompt_logprobs = []
    sampling_model_completion_logprobs = []
    target_model_prompt_logprobs = []
    target_model_completion_logprobs = []
    # KL divergence values
    kl1_seqs = []  # KL of sequence
    kl1_tokens = []  # mean KL of tokens
    kl3_seqs = []
    kl3_tokens = []

    batch_size = config.compute_kl_config.batch_size
    num_samples = len(chunk)
    for i in tqdm(range(0, num_samples, batch_size)):
        # Select a mini-batch from the chunk
        batch = chunk.select(range(i, min(i + batch_size, num_samples)))

        # Extract only the assistant's response for KL computation
        chats = batch[chat_key]
        prompts = [[chats[0]] for chats in chats]

        # Tokenize completions
        tokenized_chats = tokenizer.apply_chat_template(
            chats,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            truncation=True,
            max_length=config.compute_kl_config.max_seq_length,
        )
        tokenized_prompts = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            truncation=True,
            max_length=config.compute_kl_config.max_seq_length,
        )

        input_ids = tokenized_chats["input_ids"].to("cuda:0")
        attention_mask = tokenized_chats["attention_mask"].to("cuda:0")
        prompt_attention_mask = tokenized_prompts["attention_mask"].to("cuda:0")

        with torch.no_grad():
            s_logits = sampling_model(input_ids, attention_mask=attention_mask).logits
            t_logits = target_model(input_ids, attention_mask=attention_mask).logits
        labels = input_ids[:, 1:].contiguous()
        s_logits = s_logits[:, :-1, :].contiguous()
        t_logits = t_logits[:, :-1, :].contiguous()
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        s_log_probs = s_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        t_log_probs = t_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        # Compute KL divergence estimators http://joschu.net/blog/kl-approx.html
        logr = t_log_probs - s_log_probs
        kl1 = -logr
        kl3 = (logr.exp() - 1) - logr

        # Separate prompt and completion log probs
        for j in range(len(prompt_attention_mask)):
            last_prompt_token = prompt_attention_mask[j].sum() - 1
            last_completion_token = attention_mask[j].sum() - 1

            s_prompt_log_probs = s_log_probs[
                j, :last_prompt_token
            ]  # last token predicted by the one before
            t_prompt_log_probs = t_log_probs[j, :last_prompt_token]
            s_completions_log_probs = s_log_probs[
                j, last_prompt_token:last_completion_token
            ]
            t_completions_log_probs = t_log_probs[
                j, last_prompt_token:last_completion_token
            ]

            sampling_model_prompt_logprobs.append(s_prompt_log_probs.cpu().tolist())
            target_model_prompt_logprobs.append(t_prompt_log_probs.cpu().tolist())
            sampling_model_completion_logprobs.append(
                s_completions_log_probs.cpu().tolist()
            )
            target_model_completion_logprobs.append(
                t_completions_log_probs.cpu().tolist()
            )

            kl1_seqs.append(
                kl1[j, last_prompt_token:last_completion_token].sum().cpu().item()
            )
            kl1_tokens.append(
                kl1[j, last_prompt_token:last_completion_token].mean().cpu().item()
            )
            kl3_seqs.append(
                kl3[j, last_prompt_token:last_completion_token].sum().cpu().item()
            )
            kl3_tokens.append(
                kl3[j, last_prompt_token:last_completion_token].mean().cpu().item()
            )

    return (
        kl1_seqs,
        kl1_tokens,
        kl3_seqs,
        kl3_tokens,
        sampling_model_prompt_logprobs,
        target_model_prompt_logprobs,
        sampling_model_completion_logprobs,
        target_model_completion_logprobs,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run-online-eval",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    random.seed(config.seed)

    base_model_path = config.model_args.model_name_or_path
    training_dir = Path(config.training_dir) / "checkpoints"
    resuming_dirs = sorted(training_dir.glob("*"))
    assert (
        len(resuming_dirs) == 1
    ), f"Found {len(resuming_dirs)} directories in {training_dir}, expected 1"
    training_dir = resuming_dirs[0]

    if not (training_dir / "checkpoint-0").exists():
        relative_path = os.path.relpath(base_model_path, training_dir)
        (training_dir / "checkpoint-0").symlink_to(
            relative_path, target_is_directory=True
        )

    # Get checkpoints.
    checkpoints = sorted(
        Path(training_dir).glob("checkpoint-*"),
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )
    if config.debug_subsample > 0:
        checkpoints = checkpoints[: config.debug_subsample]

    model_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    seeds = range(config.num_seeds)

    for seed in seeds:
        # Load the eval dataset.
        if config.eval_type == "eval_split":
            ref_dataset = datasets.load_from_disk(config.dataset_args.dataset_path)[
                config.dataset_args.eval_split.name
            ]
            get_prompt_from_ref_dataset_row = lambda row: row["chosen"][0]["content"]
        elif config.eval_type == "alpaca_eval":
            ref_dataset = datasets.load_from_disk(
                Path(config.data_dir) / "shared/datasets/alpaca"
            )["eval"]
            get_prompt_from_ref_dataset_row = lambda row: row["instruction"]
        else:
            raise ValueError(f"Unsupported eval type: {config.eval_type}")

        # --- vLLM Generation (using subprocesses) ---
        cached_generation_prompts = None
        for checkpoint in checkpoints:
            evals_dir = (
                checkpoint
                / f"{config.custom_eval_prefix}online_evals_{seed}"
                / f"{config.eval_type}"
            )
            evals_dir.mkdir(parents=True, exist_ok=True)

            completions_dataset_path = evals_dir / f"completions-dataset"
            if completions_dataset_path.exists():
                if config.eval_type == "alpaca_eval":
                    alpaca_outputs_path = evals_dir / "model_outputs.json"
                    if alpaca_outputs_path.exists():
                        logger.info(
                            f"Skipping generation as {completions_dataset_path} and {alpaca_outputs_path} already exist."
                        )
                        continue
                else:
                    logger.info(
                        f"Skipping generation as {completions_dataset_path} already exists."
                    )
                    continue

            logger.info(f"Start completions generation for checkpoint {checkpoint}")

            if cached_generation_prompts is None:
                generation_prompts = [
                    model_tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": get_prompt_from_ref_dataset_row(row),
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for row in ref_dataset
                ]
                cached_generation_prompts = generation_prompts
            else:
                generation_prompts = cached_generation_prompts

            num_parts = 4
            chunk_size = len(generation_prompts) // num_parts
            chunks = []
            for i in range(num_parts):
                start = i * chunk_size
                end = (
                    (i + 1) * chunk_size
                    if i < num_parts - 1
                    else len(generation_prompts)
                )
                chunks.append(generation_prompts[start:end])

            sampling_params_dict = {
                "temperature": config.model_eval_generation_config.temperature,
                "top_p": config.model_eval_generation_config.top_p,
                "n": 1,
                "max_tokens": config.generate_eval_completions.max_new_tokens,
                "seed": seed,
            }
            checkpoint_path_str = str(checkpoint.absolute())
            with ProcessPoolExecutor(max_workers=num_parts) as executor:
                outputs_chunks = list(
                    executor.map(
                        generate_chunk_worker,
                        [checkpoint_path_str] * num_parts,
                        chunks,
                        [config.model_args.torch_dtype] * num_parts,
                        [sampling_params_dict] * num_parts,
                        list(range(num_parts)),
                    )
                )
            sleep(10)
            outputs = [output for chunk in outputs_chunks for output in chunk]

            generations_dataset = datasets.Dataset.from_dict(
                {
                    "prompt_and_completion": [
                        [
                            {
                                "role": "user",
                                "content": get_prompt_from_ref_dataset_row(row),
                            },
                            {"role": "assistant", "content": output},
                        ]
                        for row, output in zip(ref_dataset, outputs)
                    ],
                }
            )
            if not completions_dataset_path.exists():
                logger.info(
                    f"Completions generated and saved for checkpoint {checkpoint}: {completions_dataset_path}"
                )
                generations_dataset.save_to_disk(completions_dataset_path)

            if config.eval_type == "alpaca_eval":
                alpaca_outputs_path = evals_dir / "model_outputs.json"
                alpaca_outputs = [
                    {
                        "instruction": ref_dataset[i]["instruction"],
                        "output": output,
                        "generator": str(checkpoint),
                    }
                    for i, output in enumerate(outputs)
                ]
                if not alpaca_outputs_path.exists():
                    with open(alpaca_outputs_path, "w") as json_file:
                        json.dump(alpaca_outputs, json_file, indent=4)
                    logger.info(f"Alpaca outputs saved at {alpaca_outputs_path}")

        # --- Reward Computation (parallelized) ---
        # Process only the first checkpoint for demonstration.
        for checkpoint in checkpoints:
            rewards_dataset_path = (
                checkpoint
                / f"{config.custom_eval_prefix}online_evals_{seed}"
                / f"{config.eval_type}"
                / "rewards-dataset"
            )
            if rewards_dataset_path.exists():
                logger.info(
                    f"Skipping reward computation as {rewards_dataset_path} already exists."
                )
                continue

            completions_dataset_path = (
                checkpoint
                / f"{config.custom_eval_prefix}online_evals_{seed}"
                / f"{config.eval_type}"
                / "completions-dataset"
            )
            completions_dataset = datasets.load_from_disk(completions_dataset_path)
            dataset_len = len(completions_dataset)
            num_parts = 4
            chunk_size = dataset_len // num_parts
            reward_chunks = []
            for i in range(num_parts):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_parts - 1 else dataset_len
                reward_chunks.append(completions_dataset.select(range(start, end)))

            # Use a ProcessPoolExecutor to compute rewards in parallel, each on a dedicated GPU.
            with ProcessPoolExecutor(max_workers=num_parts) as executor:
                results = list(
                    executor.map(
                        compute_rewards_worker,
                        reward_chunks,
                        [config] * num_parts,
                        list(range(num_parts)),
                    )
                )
            # Merge results.
            all_rewards = []
            all_reward_tokens = []
            all_reward_tokens_len = []
            all_reward_texts = []
            for rewards, reward_tokens, reward_tokens_len, reward_texts in results:
                all_rewards.extend(rewards)
                all_reward_tokens.extend(reward_tokens)
                all_reward_tokens_len.extend(reward_tokens_len)
                all_reward_texts.extend(reward_texts)

            # Add computed rewards to the completions dataset.
            rewards_dataset = completions_dataset.add_column("reward", all_rewards)
            rewards_dataset = rewards_dataset.add_column(
                "reward_tokens", all_reward_tokens
            )
            rewards_dataset = rewards_dataset.add_column(
                "reward_tokens_len", all_reward_tokens_len
            )
            rewards_dataset = rewards_dataset.add_column(
                "reward_text", all_reward_texts
            )
            if not rewards_dataset_path.exists():
                rewards_dataset.save_to_disk(rewards_dataset_path)
                logger.info(
                    f"Rewards computed and saved for checkpoint {checkpoint}: {rewards_dataset_path}"
                )

        if config.eval_type == "eval_split":

            def add_ref_prompt_and_completion(row):
                chats = row["ref_completions"]
                prompt = chats[0]
                completions = json.loads(chats[1]["content"])
                # Arbitrarily select the last completion
                return {
                    "kl/ref_prompt_and_completion": [
                        prompt,
                        {"role": chats[1]["role"], "content": completions[-1]},
                    ],
                    "kl/ref_reward": row["ref_rewards"][-1],
                    "kl/ref_reward_tokens": row["ref_completions_reward_tokens"][-1],
                    "kl/ref_reward_tokens_len": row[
                        "ref_completions_reward_tokens_len"
                    ][-1],
                    "kl/ref_reward_text": row["ref_completions_reward_texts"][-1],
                }

            ref_dataset = ref_dataset.map(
                add_ref_prompt_and_completion,
                num_proc=240,
            )

        # --- Compute KL Divergence ---
        for checkpoint in checkpoints:
            kl_dataset_path = (
                checkpoint
                / f"{config.custom_eval_prefix}online_evals_{seed}"
                / f"{config.eval_type}"
                / "kl-dataset"
            )
            if kl_dataset_path.exists():
                logger.info(
                    f"Skipping KL computation as {kl_dataset_path} already exists."
                )
                continue

            rewards_dataset_path = (
                checkpoint
                / f"{config.custom_eval_prefix}online_evals_{seed}"
                / f"{config.eval_type}"
                / "rewards-dataset"
            )
            rewards_dataset = datasets.load_from_disk(rewards_dataset_path)

            if config.eval_type == "eval_split":
                kl_dataset = (
                    rewards_dataset.add_column(
                        "ref_prompt_and_completion",
                        ref_dataset["kl/ref_prompt_and_completion"],
                    )
                    .add_column(
                        "ref_reward",
                        ref_dataset["kl/ref_reward"],
                    )
                    .add_column(
                        "ref_reward_tokens",
                        ref_dataset["kl/ref_reward_tokens"],
                    )
                    .add_column(
                        "ref_reward_tokens_len",
                        ref_dataset["kl/ref_reward_tokens_len"],
                    )
                    .add_column(
                        "ref_reward_text",
                        ref_dataset["kl/ref_reward_text"],
                    )
                )
            else:
                kl_dataset = rewards_dataset

            dataset_len = len(kl_dataset)
            num_parts = 4  # Same parallelization as reward computation
            chunk_size = dataset_len // num_parts
            kl_chunks = []

            for i in range(num_parts):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_parts - 1 else dataset_len
                kl_chunks.append(kl_dataset.select(range(start, end)))

            # Use ProcessPoolExecutor to compute KL divergence in parallel
            checkpoint_path_str = str(checkpoint.absolute())
            base_model_path_str = base_model_path

            # Reverse and Forward KL divergence
            for kl_direction in ["reverse", "forward"]:
                if config.eval_type == "alpaca_eval" and kl_direction == "forward":
                    # No ref completions for alpaca_eval
                    continue

                if kl_direction == "reverse":
                    sampling_model_path = checkpoint_path_str
                    target_model_path = base_model_path_str
                    chat_key = "prompt_and_completion"
                else:
                    sampling_model_path = base_model_path_str
                    target_model_path = checkpoint_path_str
                    chat_key = "ref_prompt_and_completion"

                with ProcessPoolExecutor(max_workers=num_parts) as executor:
                    reverse_kl_results = list(
                        executor.map(
                            compute_kl_divergence_worker,
                            kl_chunks,
                            [config] * num_parts,
                            [sampling_model_path] * num_parts,
                            [target_model_path] * num_parts,
                            [chat_key] * num_parts,
                            list(range(num_parts)),
                        )
                    )

                # Merge results
                all_kl1_seqs = []
                all_kl1_tokens = []
                all_kl3_seqs = []
                all_kl3_tokens = []
                all_sampling_model_prompt_logprobs = []
                all_target_model_prompt_logprobs = []
                all_sampling_model_completion_logprobs = []
                all_target_model_completion_logprobs = []
                for (
                    kl1_seqs,
                    kl1_tokens,
                    kl3_seqs,
                    kl3_tokens,
                    sampling_model_prompt_logprobs,
                    target_model_prompt_logprobs,
                    sampling_model_completion_logprobs,
                    target_model_completion_logprobs,
                ) in reverse_kl_results:
                    all_kl1_seqs.extend(kl1_seqs)
                    all_kl1_tokens.extend(kl1_tokens)
                    all_kl3_seqs.extend(kl3_seqs)
                    all_kl3_tokens.extend(kl3_tokens)
                    all_sampling_model_prompt_logprobs.extend(
                        sampling_model_prompt_logprobs
                    )
                    all_target_model_prompt_logprobs.extend(
                        target_model_prompt_logprobs
                    )
                    all_sampling_model_completion_logprobs.extend(
                        sampling_model_completion_logprobs
                    )
                    all_target_model_completion_logprobs.extend(
                        target_model_completion_logprobs
                    )

                # Add KL divergence results to the rewards dataset
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/kl1_seqs", all_kl1_seqs
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/kl1_tokens", all_kl1_tokens
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/kl3_seqs", all_kl3_seqs
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/kl3_tokens", all_kl3_tokens
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/sampling_model_prompt_logprobs",
                    all_sampling_model_prompt_logprobs,
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/target_model_prompt_logprobs",
                    all_target_model_prompt_logprobs,
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/sampling_model_completion_logprobs",
                    all_sampling_model_completion_logprobs,
                )
                kl_dataset = kl_dataset.add_column(
                    f"{kl_direction}/target_model_completion_logprobs",
                    all_target_model_completion_logprobs,
                )
            # Save the KL dataset
            if not kl_dataset_path.exists():
                kl_dataset.save_to_disk(kl_dataset_path)
                logger.info(
                    f"KL divergences computed and saved for checkpoint {checkpoint}: {kl_dataset_path}"
                )

    # --- Logging with wandb ---
    wandb.init(
        id=training_dir.name,
        resume="allow",
        mode=config.wandb.mode,
        project=config.wandb.project,
    )
    cache_lc_lcref_coeff = dict()
    for checkpoint in checkpoints:
        seed_logs = [dict() for _ in seeds]
        for seed in seeds:
            logs = seed_logs[seed]
            results_dataset_path = (
                checkpoint
                / f"{config.custom_eval_prefix}online_evals_{seed}"
                / f"{config.eval_type}"
                / "kl-dataset"
            )
            results_dataset = datasets.load_from_disk(results_dataset_path)
            # Split the datasets in 2.
            # Find the indices for the selection subset and the indices for reporting subsection.
            if config.eval_type == "eval_split":
                # follows convention, could be done better.
                base_ds_name = Path(config.dataset_args.dataset_path).name.split("-")[
                    2:4
                ]
                keys_path = (
                    Path(config.dataset_args.dataset_path).parent.parent.parent
                    / "datasets-with-chosen-rewards/merged"
                    / "-".join(base_ds_name)
                )
                eval_select_keys = np.load(
                    keys_path / f"eval-checkpoint-selection-indices.npy"
                )
                eval_report_keys = np.load(
                    keys_path / f"eval-reward-reporting-indices.npy"
                )
                subsets_keys = [
                    eval_select_keys,
                    eval_report_keys,
                    np.arange(len(results_dataset)),
                ]
                subsets_names = ["selection", "reporting", "aggregate"]
            elif config.eval_type == "alpaca_eval":
                # no need to separate
                subsets_keys = [np.arange(len(results_dataset))]
                subsets_names = ["reporting"]
            else:
                raise ValueError(f"Unsupported eval type: {config.eval_type}")

            for subset_keys, subset_name in zip(subsets_keys, subsets_names):
                results_dataset_subset = results_dataset.select(subset_keys)
                len_subset = len(results_dataset_subset)
                logs[f"online_eval/{config.eval_type}/{subset_name}/avg_reward"] = (
                    sum(results_dataset_subset["reward"]) / len_subset
                )
                logs[
                    f"online_eval/{config.eval_type}/{subset_name}/avg_reward_tokens_len"
                ] = (sum(results_dataset_subset["reward_tokens_len"]) / len_subset)

                for kl_direction in ["reverse", "forward"]:
                    if config.eval_type == "alpaca_eval" and kl_direction == "forward":
                        continue
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/{kl_direction}/avg_kl1_seqs"
                    ] = (
                        sum(results_dataset_subset[f"{kl_direction}/kl1_seqs"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/{kl_direction}/avg_kl1_tokens"
                    ] = (
                        sum(results_dataset_subset[f"{kl_direction}/kl1_tokens"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/{kl_direction}/avg_kl3_seqs"
                    ] = (
                        sum(results_dataset_subset[f"{kl_direction}/kl3_seqs"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/{kl_direction}/avg_kl3_tokens"
                    ] = (
                        sum(results_dataset_subset[f"{kl_direction}/kl3_tokens"])
                        / len_subset
                    )

                if config.eval_type == "eval_split":
                    # Modify the dataset creation to be subset-specific
                    modified_rewards_dataset_subset_path = (
                        checkpoint
                        / f"{config.custom_eval_prefix}online_evals_{seed}"
                        / f"{config.eval_type}"
                        / f"modified-and-lc-rewards-v10-dataset-{subset_name}"
                    )

                    # Check if the subset-specific modified rewards dataset already exists
                    if not modified_rewards_dataset_subset_path.exists():
                        ref_dataset_subset = ref_dataset.select(subset_keys)
                        # Add ref_rewards to the subset
                        modified_rewards_dataset_subset = (
                            results_dataset_subset.add_column(
                                "ref_rewards",
                                [
                                    ref_dataset_subset[i]["ref_rewards"]
                                    for i in range(len(ref_dataset_subset))
                                ],
                            )
                        )

                        # For qrpo modified rewards.
                        def transform_rewards_dataset(row, mean_std_train_ref_rewards_):
                            # For qrpo modified rewards.
                            reward = row["reward"]
                            ref_rewards = torch.tensor(
                                row["ref_rewards"][
                                    : config.training_args.num_ref_rewards
                                ]
                            )
                            ref_rewards_std = ref_rewards.std(dim=0)
                            normalized_ref_rewards_std = (
                                ref_rewards_std / mean_std_train_ref_rewards_
                            )
                            num_ref_rewards = len(ref_rewards)
                            min_percentile = 1 / num_ref_rewards
                            max_percentile = (num_ref_rewards - 1) / num_ref_rewards

                            percentile_reward = (ref_rewards <= reward).float().mean()
                            log_percentile_reward = torch.log(
                                torch.clamp(percentile_reward, min=min_percentile)
                            )
                            normal_dist = torch.distributions.Normal(loc=0, scale=1)
                            normal_icdf_reward = normal_dist.icdf(
                                torch.clamp(
                                    percentile_reward,
                                    min=min_percentile,
                                    max=max_percentile,
                                )
                            )
                            scaled_normal_icdf_reward = (
                                normalized_ref_rewards_std * normal_icdf_reward
                            )

                            return {
                                "percentile_reward": percentile_reward,
                                "log_percentile_reward": log_percentile_reward,
                                "normal_icdf_reward": normal_icdf_reward,
                                "scaled_normal_icdf_reward": scaled_normal_icdf_reward,
                            }

                        # Compute reference rewards statistics using the full dataset
                        mean_std_train_ref_rewards = (
                            np.array(ref_dataset_subset["ref_rewards"])[
                                :, : config.training_args.num_ref_rewards
                            ]
                            .std(axis=1, ddof=1)
                            .mean(0)
                        )

                        modified_rewards_dataset_subset = modified_rewards_dataset_subset.map(
                            transform_rewards_dataset,
                            num_proc=240,
                            fn_kwargs={
                                "mean_std_train_ref_rewards_": mean_std_train_ref_rewards,
                            },
                        )

                        # For Length control modified rewards
                        # lc_ prefix means used to compute lc metrics.
                        lc_ref_rewards = np.array(ref_dataset_subset["ref_rewards"])
                        lc_ref_r_mean = lc_ref_rewards.mean(axis=1)
                        lc_ref_r_std = lc_ref_rewards.std(axis=1, ddof=1)

                        lc_ref_lengths = np.array(
                            ref_dataset_subset["ref_completions_reward_tokens_len"]
                        )
                        lc_ref_l_mean = lc_ref_lengths.mean(axis=1)
                        lc_ref_l_std = lc_ref_lengths.std(axis=1, ddof=1)

                        # Fit the length control coefficient for each prompt on the ref data
                        if not subset_name in cache_lc_lcref_coeff:
                            lc_lcref_coeff = np.zeros(len(ref_dataset_subset))
                            for i in range(len(ref_dataset_subset)):
                                if (lc_ref_l_std[i] * lc_ref_r_std[i]) == 0:
                                    lc_lcref_coeff[i] = 0
                                else:
                                    # Linear regression
                                    # R = coeff * length
                                    X = lc_ref_lengths[i]
                                    y = lc_ref_rewards[i]
                                    # normalize
                                    X = (X - lc_ref_l_mean[i]) / lc_ref_l_std[i]
                                    y = (y - lc_ref_r_mean[i]) / lc_ref_r_std[i]
                                    reg = LinearRegression(
                                        fit_intercept=False
                                    )  # already centered data
                                    X = X.reshape(-1, 1)
                                    reg.fit(X, y)
                                    lc_lcref_coeff[i] = reg.coef_[0]
                            cache_lc_lcref_coeff[subset_name] = lc_lcref_coeff
                        else:
                            lc_lcref_coeff = cache_lc_lcref_coeff[subset_name]

                        # Same but fit on the model data for all prompts together
                        lc_model_rewards = np.array(
                            modified_rewards_dataset_subset["reward"]
                        )
                        lc_model_lengths = np.array(
                            modified_rewards_dataset_subset["reward_tokens_len"]
                        )

                        # Remove prompts when we can't quantify k or it's 0. Will be k=0 for inference
                        lc_valid_indices = (lc_ref_l_std * lc_ref_r_std) != 0
                        if lc_valid_indices.sum() > 0:
                            # Linear regression
                            # R = coeff * length
                            X = lc_model_lengths[lc_valid_indices]
                            y = lc_model_rewards[lc_valid_indices]
                            # normalize
                            X = (X - lc_ref_l_mean[lc_valid_indices]) / lc_ref_l_std[
                                lc_valid_indices
                            ]
                            y = (y - lc_ref_r_mean[lc_valid_indices]) / lc_ref_r_std[
                                lc_valid_indices
                            ]
                            reg = LinearRegression(fit_intercept=True)
                            X = X.reshape(-1, 1)
                            reg.fit(X, y)
                            lc_lcmodel_coeff = reg.coef_[0]
                            lc_lcmodel_intercept = reg.intercept_
                        else:
                            lc_lcmodel_coeff = 0
                            lc_lcmodel_intercept = 0

                        # Compute the LC rewards
                        std_ratio = lc_ref_r_std / lc_ref_l_std
                        # Set NaNs and infs to 0, in all cases coeff = 0
                        std_ratio_masked = np.nan_to_num(
                            std_ratio, nan=0, posinf=0, neginf=0
                        )

                        # Length control with respect to ref data

                        # R = (r~ - kl~)*stdr + meanr = (r-meanr)/stdr*stdr - k (l - meanl)/stdl * stdr + meanr
                        # = r - k * (l - meanl)*stdl/stdl

                        lc_lcref_reward = (
                            lc_model_rewards
                            - lc_lcref_coeff
                            * (lc_model_lengths - lc_ref_l_mean)
                            * std_ratio_masked
                        )

                        # Length control with respect to model data
                        # recovering in the reward space.
                        lc_lcmodel_reward = (
                            lc_model_rewards
                            - lc_lcmodel_coeff
                            * (lc_model_lengths - lc_ref_l_mean)
                            * std_ratio_masked
                        )
                        # Directly in the normalized reward space
                        # lc_norm_reward = b * ref_r_std + ref_r_mu
                        lc_lcmodelprenorm_reward = (
                            lc_lcmodel_intercept * lc_ref_r_std + lc_ref_r_mean
                        )
                        lc_lcmodelprenorm_reward[~lc_valid_indices] = lc_model_rewards[
                            ~lc_valid_indices
                        ]

                        # Add to dataset
                        modified_rewards_dataset_subset = (
                            modified_rewards_dataset_subset.add_column(
                                "lc_ref_r_mean", lc_ref_r_mean
                            )
                            .add_column("lc_ref_r_std", lc_ref_r_std)
                            .add_column("lc_ref_l_mean", lc_ref_l_mean)
                            .add_column("lc_ref_l_std", lc_ref_l_std)
                            .add_column("lc_lcref_coeff", lc_lcref_coeff)
                            .add_column(
                                "lc_lcmodel_coeff",
                                [
                                    lc_lcmodel_coeff
                                    for _ in range(len(modified_rewards_dataset_subset))
                                ],
                            )
                            .add_column(
                                "lc_lcmodel_intercept",
                                [
                                    lc_lcmodel_intercept
                                    for _ in range(len(modified_rewards_dataset_subset))
                                ],
                            )
                            .add_column("lc_lcref_reward", lc_lcref_reward)
                            .add_column("lc_lcmodel_reward", lc_lcmodel_reward)
                            .add_column(
                                "lc_lcmodelprenorm_reward", lc_lcmodelprenorm_reward
                            )
                        )

                        # Save the subset-specific modified rewards dataset
                        if not modified_rewards_dataset_subset_path.exists():
                            modified_rewards_dataset_subset.save_to_disk(
                                modified_rewards_dataset_subset_path
                            )

                    else:
                        logger.info(
                            f"Skipping modified rewards computation as {modified_rewards_dataset_subset_path} already exists."
                        )
                        modified_rewards_dataset_subset = datasets.load_from_disk(
                            modified_rewards_dataset_subset_path
                        )

                    # Compute and log metrics for the subset
                    len_subset = len(modified_rewards_dataset_subset)

                    # LC rewards
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/lc_lcref_avg_coeff"
                    ] = (
                        sum(modified_rewards_dataset_subset["lc_lcref_coeff"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/lc_lcmodel_coeff"
                    ] = (
                        sum(modified_rewards_dataset_subset["lc_lcmodel_coeff"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/lc_lcmodel_intercept"
                    ] = (
                        sum(modified_rewards_dataset_subset["lc_lcmodel_intercept"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/lc_lcref_avg_reward"
                    ] = (
                        sum(modified_rewards_dataset_subset["lc_lcref_reward"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/lc_lcmodel_avg_reward"
                    ] = (
                        sum(modified_rewards_dataset_subset["lc_lcmodel_reward"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/lc_lcmodelprenorm_avg_reward"
                    ] = (
                        sum(modified_rewards_dataset_subset["lc_lcmodelprenorm_reward"])
                        / len_subset
                    )

                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/avg_percentile_reward"
                    ] = (
                        sum(modified_rewards_dataset_subset["percentile_reward"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/avg_log_percentile_reward"
                    ] = (
                        sum(modified_rewards_dataset_subset["log_percentile_reward"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/avg_normal_icdf_reward"
                    ] = (
                        sum(modified_rewards_dataset_subset["normal_icdf_reward"])
                        / len_subset
                    )
                    logs[
                        f"online_eval/{config.eval_type}/{subset_name}/avg_scaled_normal_icdf_reward"
                    ] = (
                        sum(
                            modified_rewards_dataset_subset["scaled_normal_icdf_reward"]
                        )
                        / len_subset
                    )

        # Compute mean and std over seeds.
        logs_with_std = {"train/global_step": checkpoint.name.split("-")[-1]}
        detailed_logs = {"train/global_step": checkpoint.name.split("-")[-1]}
        for k in seed_logs[0].keys():
            mean_k = f"{k}_mean"
            err_k = f"{k}_std"

            mean = np.mean([seed_logs[seed][k] for seed in seeds])
            std = np.std([seed_logs[seed][k] for seed in seeds], ddof=1)

            logs_with_std[mean_k] = mean
            logs_with_std[err_k] = std

            # Log detailed metrics
            detailed_logs[mean_k] = mean
            detailed_logs[err_k] = std
            for seed in seeds:
                detailed_logs[f"{k}_{seed}"] = seed_logs[seed][k]

        # Write to file
        with open(
            checkpoint
            / f"{config.custom_eval_prefix}online_evals_logs_{config.eval_type}.json",
            "w",
        ) as json_file:
            json.dump(logs_with_std, json_file, indent=4)
        with open(
            checkpoint
            / f"{config.custom_eval_prefix}online_evals_detailed_logs_{config.eval_type}.json",
            "w",
        ) as json_file:
            json.dump(detailed_logs, json_file, indent=4)

        wandb.log(logs_with_std)

    logger.info(f"Completed {len(checkpoints)} checkpoints.")
    logger.info(f"Last checkpoint name: {checkpoints[-1].name}")
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
