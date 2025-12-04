"""
Merges a partitioned Hugging Face datasets directory structure into
a single dataset for a specific split ("train_split" or "eval_split").

The script expects one of the following structures:

1) Default, partitioned mode (`config.partitioned_partitions=True`):
   Directories named "<start>-<end>", each containing exactly 4 sub-chunk
   folders ("0", "1", "2", "3"). Each sub-chunk folder includes a
   "checkpoints/<hash>/checkpoint-XXX" structure (where XXX is a numeric index).

   Example directory structure:
       train_split/
           0-2048/
               0/
                   checkpoints/
                       abcd123.../  Only one hash dir allowed
                           checkpoint-128/
                           checkpoint-256/
               1/
               2/
               3/
           2048-4096/
           ...

2) No-partitions mode (`config.is_partitioned=False`):
   The *split_dir* directly contains exactly 4 sub-chunk folders ("0", "1", "2", "3")
   with the same "checkpoints/<hash>/checkpoint-XXX" structure.

   Example directory structure:
       train_split/
           0/
               checkpoints/
                   abcd123.../
                       checkpoint-128/
                       checkpoint-256/
           1/
           2/
           3/
"""

import logging
import os
import re
from pathlib import Path

import datasets
import hydra
from datasets import DatasetDict, load_from_disk
from omegaconf import DictConfig

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


def find_partition_dirs(split_dir):
    """
    Return a sorted list of valid partition directories found within split_dir.

    A valid partition directory name matches the pattern '<start>-<end>' where
    <start> and <end> are integers, e.g. '2048-4096'.
    """
    partition_pattern = re.compile(r"^(\d+)-(\d+)$")
    candidates = [
        fname
        for fname in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, fname))
        and partition_pattern.match(fname)
    ]

    def extract_partition_start(dirname):
        match = partition_pattern.match(dirname)
        return int(match.group(1))

    # Sort by numeric start value
    candidates.sort(key=extract_partition_start)
    return candidates


def validate_chunks(partition_path):
    """
    Ensures that the partition directory contains exactly 4 sub-chunk folders: '0', '1', '2', '3'.
    Raises an error if any chunk is missing or extra chunks are found.
    """
    required_chunks = {"0", "1", "2", "3"}  # Hard coded 4 tasks per partition.
    found_chunks = {
        d
        for d in os.listdir(partition_path)
        if os.path.isdir(os.path.join(partition_path, d))
    }

    if found_chunks != required_chunks:
        raise ValueError(
            f"Partition {partition_path} must contain exactly 4 sub-chunks: {required_chunks}, "
            f"but found: {found_chunks}"
        )


def find_checkpoint_dirs(chunk_path):
    """
    Under a given chunk directory, find the single hash directory inside:
    chunk_path/checkpoints/<hash>/checkpoint-xxx

    Returns a list of absolute paths to each 'checkpoint-XXX' directory.
    Raises an error if there is not exactly one hash directory.
    """
    checkpoints_root = os.path.join(chunk_path, "checkpoints")
    if not os.path.isdir(checkpoints_root):
        raise ValueError(f"Missing 'checkpoints' directory in {chunk_path}")

    # Find hash directory (must be exactly one)
    hash_dirs = [
        d
        for d in os.listdir(checkpoints_root)
        if os.path.isdir(os.path.join(checkpoints_root, d))
    ]
    if len(hash_dirs) == 0:
        raise ValueError(f"Missing hash directory in {checkpoints_root}")
    elif len(hash_dirs) > 1:
        raise ValueError(
            f"Multiple hash directories found in {checkpoints_root}: {hash_dirs}"
        )

    hash_path = os.path.join(checkpoints_root, hash_dirs[0])

    # Find checkpoint directories
    checkpoint_dirs = [
        os.path.join(hash_path, d)
        for d in os.listdir(hash_path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(hash_path, d))
    ]

    # Sort by numeric suffix in "checkpoint-XXX"
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[1]))

    return checkpoint_dirs


def load_partition_datasets(partition_path, is_subpartitioned):
    """
    Validate 4 sub-chunks and load them. Returns a list of loaded datasets.
    """
    loaded = []
    if is_subpartitioned:
        validate_chunks(partition_path)
        # Iterate through the 4 required sub-chunk folders
        for subpartition in range(4):
            subpartition_folder = os.path.join(partition_path, str(subpartition))
            # Collect all checkpoint directories and load them
            checkpoint_paths = find_checkpoint_dirs(subpartition_folder)
            for ckpt_dir in checkpoint_paths:
                ds = load_from_disk(ckpt_dir)
                logger.info(f"Loaded dataset from: {ckpt_dir}")
                loaded.append(ds)
    else:
        checkpoint_paths = find_checkpoint_dirs(partition_path)
        for ckpt_dir in checkpoint_paths:
            ds = load_from_disk(ckpt_dir)
            logger.info(f"Loaded dataset from: {ckpt_dir}")
            loaded.append(ds)

    return loaded


def merge_split(config, split_name, split_dir):
    """
    Merges all partitioned (or no-partition) datasets found under `split_dir` into
    a single dataset, and returns the result.

    Parameters:
    -----------
    split_dir  : str
        Directory containing either the partitioned or the no-partition dataset sub-chunks.
    """
    # If is_partitioned=False, we treat split_dir as one "partition" containing 4 sub-chunks.
    if not config.is_partitioned:
        # Validate that there are no separate partition directories,
        # and the split_dir directly contains 0..3 sub-chunk folders.
        # So we just load from split_dir as if it's one big partition.
        logger.info(
            f"`is_partitioned=False`: treating {split_dir} as a single partition."
        )
        all_datasets = load_partition_datasets(
            split_dir, is_subpartitioned=config.is_subpartitioned
        )

    else:
        # Partition-based logic:
        partition_dirs = find_partition_dirs(split_dir)
        if not partition_dirs:
            raise ValueError(f"No valid partition directories found in {split_dir}")

        all_datasets = []
        for partition_name in partition_dirs:
            partition_path = os.path.join(split_dir, partition_name)
            # Load sub-chunks for this partition
            part_ds_list = load_partition_datasets(
                partition_path, is_subpartitioned=config.is_subpartitioned
            )
            all_datasets.extend(part_ds_list)

    if len(all_datasets) == 0:
        raise ValueError(f"No checkpoint datasets discovered in {split_dir}")

    logging.info(f"Concatenating {len(all_datasets)} datasets ...")
    dataset = datasets.concatenate_datasets(all_datasets)

    # Confirm we have the expected number of examples
    split_size = getattr(config.dataset_args, split_name).end
    if len(dataset) != split_size:
        raise ValueError(
            f"Expected {split_size} examples in {split_name}, but found {len(dataset)}"
        )

    return dataset


@hydra.main(version_base=None, config_path="../configs", config_name="merge-partitions")
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)

    # If the dataset already exists in save path, exit
    if os.path.exists(config.save_path):
        logger.info(f"Dataset already exists in {config.save_path}. Exiting.")
        return

    # Merge both 'train_split' and 'eval_split' by default
    splits = dict()
    for split in ["train_split", "eval_split"]:
        split_dir = Path(config.dataset_path) / split
        split_name = getattr(config.dataset_args, split).name
        try:
            splits[split_name] = merge_split(config, split, split_dir)
        except Exception as e:
            logger.error(f"Error: could not merge {split}: {e}")
            raise e

    # Merge the splits and save as a HF dataset
    dataset = DatasetDict(splits)
    logger.info(f"Saving merged dataset to {config.save_path}")
    dataset.save_to_disk(config.save_path)
    logger.info("Dataset saved successfully.")


if __name__ == "__main__":
    main()
