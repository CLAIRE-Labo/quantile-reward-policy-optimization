import logging
from pathlib import Path

import hydra
import numpy as np
from datasets import load_from_disk
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="create-eval-split",
)
def main(config: DictConfig) -> None:
    eval_dataset = load_from_disk(config.dataset_args.dataset_path)[
        config.dataset_args.eval_split.name
    ]

    # Convert the Hugging Face Dataset to a pandas DataFrame
    df = eval_dataset.to_pandas()

    if "leetcode" in Path(config.dataset_args.dataset_path).name:
        df["stratify_col"] = (
            df[
                [
                    "difficulty",
                ]
            ]
            .astype(str)
            .agg("_".join, axis=1)
        )

        # Count occurrences of each stratification group
        group_counts = df["stratify_col"].value_counts()

        # Filter out groups with fewer than 2 samples
        valid_classes = group_counts[group_counts >= 2].index
        df_filtered = df[df["stratify_col"].isin(valid_classes)]

        # Get the indices for the filtered DataFrame
        indices = df_filtered.index

        # Perform a stratified split (50/50) using the filtered data
        part1_idx, part2_idx = train_test_split(
            indices,
            test_size=config.reward_reporting_data_fraction,
            random_state=config.seed,  # ensures reproducibility
            stratify=df_filtered["stratify_col"],
        )
    else:
        # Get the indices for the DataFrame
        indices = df.index

        # Perform a non-stratified split (50/50)
        part1_idx, part2_idx = train_test_split(
            indices,
            test_size=config.reward_reporting_data_fraction,
            random_state=config.seed,  # ensures reproducibility
        )

    # Save the indices into two separate files
    np.save(config.eval_reward_reporting_indexes_path, part1_idx)
    np.save(config.eval_checkpoint_selection_indexes_path, part2_idx)

    logger.info(
        f"Checkpoint selection indices saved to {config.eval_checkpoint_selection_indices_path}"
    )
    logger.info(
        f"Reward reporting indices saved to {config.eval_reward_reporting_indices_path}"
    )


if __name__ == "__main__":
    main()
