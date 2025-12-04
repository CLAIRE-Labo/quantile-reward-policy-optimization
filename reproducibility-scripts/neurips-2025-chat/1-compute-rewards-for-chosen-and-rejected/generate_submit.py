from datetime import datetime
from pathlib import Path

import yaml

"""Nomenclature:

dataset = f"{dataset}"
reward_model = f"{reward_model}"

dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
"""

stdout_prefix = "all"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)
job_name = "datasets-with-chosen-rewards"

datasets = ["magpieair", "ultrafeedback"]
splits = ["train_split", "eval_split"]
reward_models = ["armorm"]

save_interval = 8192
num_nodes = 1

commands = []
nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for split in splits:
            dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
            jobid = f"{dataset_with_chosen_rewards}/{split}"

            with open(f"src/qrpo/configs/dataset/{dataset}.yaml", "r") as file:
                dataset_config = yaml.safe_load(file)
            split_config = dataset_config["dataset_args"][split]
            split_name = split_config["name"]
            split_size = split_config["end"] - split_config["start"]

            commands.append(
                (
                    "sbatch "
                    f"-N {num_nodes} "
                    f"-o {stdout_root}/out/{jobid}.out "
                    f"-e {stdout_root}/out/{jobid}.err "
                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-compute-rewards-for-chosen-and-rejected.sh "
                    f"dataset={dataset} "
                    f"reward_model={reward_model} "
                    f"split={split_name} "
                    f"save_interval={save_interval} "
                    f"job_subdir_prefix={job_name}/{jobid} "
                    "outputs_subdir=shared "
                    "resuming.resume=True "
                )
            )
            nodes_needed += num_nodes

# Write th submit commands to a new directory where this batch of experiments will be managed)
# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Needed nodes:", nodes_needed)
