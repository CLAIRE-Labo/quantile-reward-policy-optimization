import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import datasets
import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


# Function to generate completions in batches
def generate_completions_batch(batch, llm, tokenizer, config):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in batch
    ]

    if getattr(config.ref_completion_generation, "extra_stop_token_id") is not None:
        stop_token_ids = [
            tokenizer.eos_token_id,
            config.ref_completion_generation.extra_stop_token_id,
        ]
    else:
        stop_token_ids = [tokenizer.eos_token_id]

    sampling_params = SamplingParams(
        temperature=config.ref_completion_generation.temperature,
        top_p=config.ref_completion_generation.top_p,
        n=1,
        max_tokens=config.max_tokens,
        stop_token_ids=stop_token_ids,
    )

    # Generate completions
    outputs = llm.generate(prompts, sampling_params)

    return [
        {
            "instruction": batch[i]["instruction"],
            "output": output.outputs[0].text.strip(),
            "generator": config.model_name,
        }
        for i, output in enumerate(outputs)
    ]


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run-alpaca-eval-vllm",
)
def main(config: DictConfig) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        logger.error(
            "Error: OPENAI_API_KEY is not set. Please set it before running the script."
        )
        sys.exit(1)

    random.seed(config.seed)

    # Ensure output directory exists
    model_path = Path(config.outputs_dir).parent / config.model_args.model_name_or_path
    output_dir = model_path / "alpaca_eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    completions_file = output_dir / "model_outputs.json"

    # Check if output file already exists
    if completions_file.exists():
        logger.info(f"Skipping generation as {completions_file} already exists.")
    else:
        logger.info(f"Start completions generation for alpaca eval 2")
        data = datasets.load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True
        )["eval"]

        llm = LLM(
            str(model_path.absolute()),
            dtype=config.model_args.torch_dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=0.8,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path.absolute())
        model_outputs = generate_completions_batch(data, llm, tokenizer, config)
        with open(completions_file, "w") as json_file:
            json.dump(model_outputs, json_file, indent=4)

        logger.info(f"Completions generated and saved at {completions_file}")

    # Evaluate completions
    logger.info(f"Start evaluation")
    alpaca_output_path = output_dir / "alpaca_eval_gpt4_turbo_fn"
    if alpaca_output_path.exists():
        logger.info(
            f"Skipping Alpaca Eval 2: Results already exist at {alpaca_output_path}"
        )
    else:
        logger.info("Running Alpaca Eval 2...")
        cmd_alpaca_eval = [
            "alpaca_eval",
            "--model_outputs",
            str(completions_file),
            "--annotators_config",
            "alpaca_eval_gpt4_turbo_fn",
            "--output_path",
            str(output_dir.absolute()),
        ]
        logger.info("Running command:", " ".join(cmd_alpaca_eval))
        subprocess.run(cmd_alpaca_eval, check=True)

    logger.info(f"Pipeline completed. Results are in {output_dir}")

    # Read the alpaca eval results csv
    csv_file = alpaca_output_path / "leaderboard.csv"
    if not csv_file.exists():
        logger.error(f"Error: Alpaca Eval 2 results not found at {csv_file}")
        sys.exit(1)

    # Read the csv file
    # looks like this:
    alpaca_results = pd.read_csv(csv_file)

    # log to wandb
    wandb.init(
        id=config.wandb.run_id,
        resume="allow",
        mode=config.wandb.mode,
        project=config.wandb.project,
    )

    # Log the csv at the step of the checkpoint
    wandb.log(
        {
            "train/global_step": config.checkpoint_step,
            f"alpaca/evals_at_{config.checkpoint_step}": wandb.Table(
                dataframe=alpaca_results
            ),
            **{f"alpaca/{k}": v for k, v in dict(alpaca_results.iloc[0]).items()},
        }
    )


if __name__ == "__main__":
    main()
