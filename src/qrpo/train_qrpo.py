import logging
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import hydra
import numpy as np
import torch
import wandb
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qrpo import utils
from qrpo.trainers.qrpo import QRPOTrainer
from qrpo.trainers.qrpo_config import QRPOConfig
from qrpo.utils import utils_for_trl

utils.config.register_resolvers()
acc_state = PartialState(
    **accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=4)).to_kwargs()
)
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train-qrpo")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################

    config = utils_for_trl.setup_config_and_resuming(config, acc_state, acc_logger)
    # full_config is a merge with the TRL arg dataclasses
    # The args dataclasses are used by the HF classes, and the full_config by the template.
    script_args = ScriptArguments(**OmegaConf.to_container(config.script_args))
    training_args = QRPOConfig(
        **OmegaConf.to_container(config.training_args), output_dir=str(Path.cwd())
    )
    model_args = ModelConfig(**OmegaConf.to_container(config.model_args))

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    full_config = utils_for_trl.merge_and_save_config(
        config, script_args, training_args, model_args, acc_state
    )
    if acc_state.is_main_process:
        utils.config.setup_wandb(full_config, acc_logger)
    utils.seeding.seed_everything(config)

    ############################ Tokenizer Setup ############################

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Update special tokens.
    if getattr(config.tokenizer_args, "pad_token_id") is not None:
        if config.tokenizer_args.pad_token_id != tokenizer.pad_token_id:
            acc_logger.info(f"Found tokenizer pad token id: {tokenizer.pad_token_id}")
            acc_logger.info(
                f"But config specifies a different pad token id: {config.tokenizer_args.pad_token_id}"
            )
            tokenizer.pad_token_id = config.tokenizer_args.pad_token_id
            acc_logger.info(
                f"Overriding tokenizer pad token id to {config.tokenizer_args.pad_token_id}"
            )

    # Perform checks
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must have a pad token.")
    if tokenizer.pad_token == tokenizer.eos_token:
        raise ValueError(
            "Tokenizer pad token is the same as the eos token. The eos will be masked as if it was a pad."
        )

    ############################ Dataset Setup ############################

    with acc_state.main_process_first():
        ds = datasets.load_from_disk(script_args.dataset_name)
        ds = DatasetDict(
            {
                "train": ds[config.script_args.dataset_train_split],
                "eval": ds[config.script_args.dataset_test_split],
            }
        )
        # We only support the "preference with implicit prompt" format
        # with "chosen" and "rejected" columns including both chat and ref completions
        # https://huggingface.co/docs/trl/main/en/dataset_formats#preference
        # Drop the extra preference columns
        for extra_key in ["prompt", "completion", "messages", "label"]:
            if extra_key in ds["train"].column_names:
                ds["train"] = ds["train"].remove_columns([extra_key])
            if extra_key in ds["eval"].column_names:
                ds["eval"] = ds["eval"].remove_columns([extra_key])

        if config.dataset_args.debug_oom:
            with acc_state.main_process_first():
                ds = ds.sort("max_chosen_rejected_reward_tokens_len", reverse=True)

        if config.dataset_args.debug_subsample.train > 0:
            ds["train"] = ds["train"].select(
                range(min(len(ds["train"]), config.dataset_args.debug_subsample.train))
            )
        if config.dataset_args.debug_subsample.eval > 0:
            ds["eval"] = ds["eval"].select(
                range(min(len(ds["eval"]), config.dataset_args.debug_subsample.eval))
            )

        # Shuffle at the end to preserve previous cache across seeds.
        ds = ds.shuffle(seed=config.seed)

        # Extract ref rewards and drop ref completions and rewards columns
        train_ref_rewards = np.array(ds["train"]["ref_rewards"])[
            :, : training_args.num_ref_rewards
        ]
        eval_ref_rewards = np.array(ds["eval"]["ref_rewards"])[
            :, : training_args.num_ref_rewards
        ]
        ds["train"] = ds["train"].remove_columns(["ref_rewards", "ref_completions"])
        ds["eval"] = ds["eval"].remove_columns(["ref_rewards", "ref_completions"])

        # put truncated ref_rewards back to the datasets
        mean_std_ref_rewards = train_ref_rewards.std(axis=1, ddof=1).mean(0)

        ds["train"] = ds["train"].add_column(
            name="ref_rewards", column=train_ref_rewards.tolist()
        )
        ds["eval"] = ds["eval"].add_column(
            name="ref_rewards", column=eval_ref_rewards.tolist()
        )

    ############################ Trainer Setup ############################

    # Find the last checkpoint
    resuming_dir = Path.cwd()
    last_checkpoint_number = max(
        (
            int(item.name.split("-")[-1])
            for item in resuming_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint-")
        ),
        default=0,
    )
    # Handle resuming
    last_checkpoint_number = 0
    for item in resuming_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            if (item / "scheduler.pt").is_file() and (
                item / "trainer_state.json"
            ).is_file():
                last_checkpoint_number = max(
                    last_checkpoint_number, int(item.name.split("-")[-1])
                )

    if last_checkpoint_number > 0:
        acc_logger.info(
            f"TRL will attempt to resume from last checkpoint: {last_checkpoint_number}"
        )
        eval_file = resuming_dir / f"eval_{last_checkpoint_number}_results.json"
        if eval_file.exists():
            training_args.eval_on_start = False

    trainer = QRPOTrainer(
        model_args.model_name_or_path,
        ref_model=None,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        mean_std_ref_rewards=mean_std_ref_rewards,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint_number > 0)
    acc_logger.info("Training completed. Performing final evaluation.")

    last_eval_file = resuming_dir / f"eval_results.json"
    if training_args.eval_strategy != "no":
        if last_eval_file.exists():
            acc_logger.info("Last evaluation already performed.")
        else:
            torch.cuda.empty_cache()
            acc_logger.info("Performing final evaluation.")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            acc_logger.info("Final evaluation completed.")

    acc_state.wait_for_everyone()
    acc_logger.info("Training completed. Checkpoints saved.")
    if acc_state.is_main_process:
        wandb.finish()

    acc_state.wait_for_everyone()
    accelerate.Accelerator().end_training()


if __name__ == "__main__":
    main()
