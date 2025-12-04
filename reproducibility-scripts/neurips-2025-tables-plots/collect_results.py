import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def gather_runs(
    base_dir: Path,
    job_root_subdirs,
    collect_alpaca=False,
):
    # First, gather all training_dir paths (with their corresponding job_root_subdir)
    all_training_dirs = []
    for job_root_subdir in job_root_subdirs:
        job_root_subdir_path = base_dir / job_root_subdir
        if not job_root_subdir_path.is_dir():
            print(
                f"Warning: {job_root_subdir_path.resolve()} does not exist or is not a directory. Skipping..."
            )
            continue

        for training_dir in sorted(job_root_subdir_path.iterdir()):
            if training_dir.is_dir():
                all_training_dirs.append((job_root_subdir, training_dir))
            else:
                print(
                    f"Warning: {training_dir.resolve()} is not a directory. Skipping..."
                )

    rows = []

    # Now iterate over all discovered training directories with a single progress bar
    for job_root_subdir, training_dir in tqdm(
        all_training_dirs, desc="Collecting runs"
    ):
        checkpoints_path = training_dir / "checkpoints"
        if not checkpoints_path.is_dir():
            print(
                f"Warning: {checkpoints_path.resolve()} does not exist or is not a directory. Skipping..."
            )
            continue

        # Each training run typically has exactly one hash directory under 'checkpoints'.
        hash_dirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
        if not hash_dirs:
            print(
                f"Warning: no hash directories under {checkpoints_path.resolve()}. Skipping..."
            )
            continue
        if len(hash_dirs) > 1:
            print(
                f"Warning: found more than one hash under {checkpoints_path.resolve()}. Skipping entire training_dir."
            )
            continue

        hash_dir = hash_dirs[0]
        # We require a config file at config/process-0/full-config-trl.yaml
        config_file = hash_dir / "config" / "process-0" / "full-config-trl.yaml"
        if not config_file.is_file():
            raise FileNotFoundError(
                f"Could not find {config_file.resolve()} for training run {training_dir.resolve()}. Aborting."
            )

        # Parse the config, flatten at most 2 levels
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        flattened_config = {}
        for top_key, top_val in config_data.items():
            if isinstance(top_val, dict):
                for sub_key, sub_val in top_val.items():
                    # Patch to translate the previous configs to the new ones
                    if (
                        top_key == "training_args"
                    ) and sub_key == "qrpo_transform_type":
                        if sub_val == "log_mae":
                            flattened_config[
                                "training_args.qrpo_transform_type"
                            ] = "log"
                            flattened_config["training_args.qrpo_loss_type"] = "mae"
                            flattened_config[
                                "training_args.qrpo_sample_selector"
                            ] = "both"
                        elif sub_val == "log_chosen":
                            flattened_config[
                                "training_args.qrpo_transform_type"
                            ] = "log"
                            flattened_config["training_args.qrpo_loss_type"] = "mse"
                            flattened_config[
                                "training_args.qrpo_sample_selector"
                            ] = "chosen"
                        else:
                            flattened_config[
                                "training_args.qrpo_transform_type"
                            ] = sub_val

                            # set default loss type to mse, can be overridden later if present in config
                            if not "training_args.qrpo_loss_type" in flattened_config:
                                flattened_config["training_args.qrpo_loss_type"] = "mse"

                            # set default sample selector to both, can be overridden later if present in config
                            if (
                                not "training_args.qrpo_sample_selector"
                                in flattened_config
                            ):
                                flattened_config[
                                    "training_args.qrpo_sample_selector"
                                ] = "both"
                    else:
                        flattened_config[f"{top_key}.{sub_key}"] = sub_val
            else:
                flattened_config[top_key] = top_val

        # Now gather checkpoint metrics from checkpoint-* directories
        for cp_dir in sorted(hash_dir.glob("checkpoint-*")):
            # We follow symlinks; just confirm it's a directory or symlink
            if not (cp_dir.is_dir() or cp_dir.is_symlink()):
                print(
                    f"Warning: {cp_dir.resolve()} is neither a directory nor a symlink. Skipping..."
                )
                continue
            cp_dir_resolved = cp_dir.resolve()
            if not cp_dir_resolved.exists():
                print(f"Warning: symlink {cp_dir.resolve()} is broken. Skipping...")
                continue

            # Extract checkpoint number from folder name
            checkpoint_name = cp_dir.name
            if not checkpoint_name.startswith("checkpoint-"):
                print(
                    f"Warning: {cp_dir.resolve()} doesn't match 'checkpoint-*' pattern. Skipping..."
                )
                continue
            try:
                checkpoint_num = int(checkpoint_name.split("-", 1)[1])
            except ValueError:
                checkpoint_num = checkpoint_name.split("-", 1)[1]

            eval_json = cp_dir_resolved / "online_evals_detailed_logs_eval_split.json"
            if not eval_json.is_file():
                print(f"Warning: {eval_json.resolve()} is not present. Skipping...")
                continue

            with open(eval_json, "r") as f:
                eval_data = json.load(f)

            # alpaca json
            if collect_alpaca:
                eval_json = (
                    cp_dir_resolved / "online_evals_detailed_logs_alpaca_eval.json"
                )
                if not eval_json.is_file():
                    print(f"Warning: {eval_json.resolve()} is not present. Skipping...")
                    continue
                with open(eval_json, "r") as f:
                    eval_data.update(json.load(f))

                # alpaca gpt json
                alpaca_gpt_json = cp_dir_resolved / "alpaca_eval_gpt4_turbo_fn.json"
                if alpaca_gpt_json.is_file():
                    with open(alpaca_gpt_json, "r") as f:
                        alpaca_data = json.load(f)
                        # Convert anything that can be converted to a float
                        for key, value in alpaca_data.items():
                            try:
                                alpaca_data[key] = float(value)
                            except (ValueError, TypeError):
                                alpaca_data[key] = value
                        eval_data.update(alpaca_data)

            row = {
                "job_root_subdir": job_root_subdir,
                "training_dir": training_dir.name,
                "hash": hash_dir.name,
                "checkpoint_num": checkpoint_num,
            }
            # Merge flattened config and evaluation metrics
            row.update(flattened_config)
            row.update(eval_data)

            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    job_name_prefix = "neurips-2025"
    job_root_subdirs = ["chat-baseline", "chat-scaling", "code-baseline", "code-nosft"]

    for job_root_subdir in job_root_subdirs:
        job_name = f"{job_name_prefix}-{job_root_subdir}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print("Crawling directories to build DataFrame...")
        base_dir = Path("outputs/shared/train_qrpo")
        if job_root_subdir == "chat-baseline":
            collect_alpaca = True
        else:
            collect_alpaca = False
        df = gather_runs(
            base_dir,
            [job_root_subdir],
            collect_alpaca=collect_alpaca,
        )

        # Ensure our output directory exists, then save
        out_path_csv = Path("outputs/shared/parsed-results") / f"{job_name}.csv"
        out_path_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path_csv, index=False)
        print(f"DataFrame saved to: {out_path_csv.resolve()}")
