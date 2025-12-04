import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run-alpaca-gpt",
)
def main(config: DictConfig) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        logger.error(
            "Error: OPENAI_API_KEY is not set. Please set it before running the script."
        )
        sys.exit(1)

    random.seed(config.seed)

    # Ensure output directory exists
    checkpoint_path = Path(config.checkpoint_path)
    completions_file = checkpoint_path / "online_evals_0/alpaca_eval/model_outputs.json"

    # Check if output file already exists
    if not completions_file.exists():
        logger.error(f"Error: Completions file not found at {completions_file}")
        sys.exit(1)

    # Evaluate completions
    logger.info(f"Start evaluation")
    alpaca_output_path = checkpoint_path / "alpaca_eval_gpt4_turbo_fn"
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
            str(checkpoint_path.absolute()),
        ]
        logger.info("Running command:", " ".join(cmd_alpaca_eval))
        subprocess.run(cmd_alpaca_eval, check=True)

    logger.info(f"Alpaca Eval queries completed. Results are in {checkpoint_path}")

    # Log to wandb.
    # Read the alpaca eval results csv
    csv_file = alpaca_output_path / "leaderboard.csv"
    if not csv_file.exists():
        logger.error(f"Error: Alpaca Eval 2 results not found at {csv_file}")
        sys.exit(1)

    alpaca_results = pd.read_csv(csv_file)
    if checkpoint_path.parent.name == "checkpoints":
        wandb_run_id = checkpoint_path.name
        global_step = 0
    else:
        wandb_run_id = checkpoint_path.parent.name
        global_step = checkpoint_path.name.split("-")[-1]
    wandb.init(
        id=wandb_run_id,
        resume="allow",
        mode=config.wandb.mode,
        project=config.wandb.project,
    )

    # Log the csv at the step of the checkpoint
    logs = {f"alpaca/{k}": str(v) for k, v in dict(alpaca_results.iloc[0]).items()}
    logs["train/global_step"] = global_step
    # log logs to file as json, from dict
    logs_path = checkpoint_path / "alpaca_eval_gpt4_turbo_fn.json"
    with open(logs_path, "w") as f:
        json.dump(logs, f)
    # log to wandb
    wandb.log(
        {
            f"alpaca/evals_at_{global_step}": wandb.Table(dataframe=alpaca_results),
            **logs,
        }
    )

    logger.info(f"Alpaca Eval pipeline completed. Results are in {checkpoint_path}")


if __name__ == "__main__":
    main()
