from datetime import datetime
from pathlib import Path

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


datasets = ["magpieair", "ultrafeedback"]
reward_models = ["armorm"]
is_partitioned = False
dataset_type = "datasets-with-chosen-rewards"
num_nodes = 1

commands = []
nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
        jobid = dataset_with_chosen_rewards
        commands.append(
            (
                "sbatch "
                f"-N {num_nodes} "
                f"-o {stdout_root}/out/{jobid}.out "
                f"-e {stdout_root}/out/{jobid}.err "
                "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended.sh "
                f"python -m qrpo.generation.merge_partitions "
                f"dataset={dataset_with_chosen_rewards} "
                f"dataset_id={dataset_with_chosen_rewards} "
                f"dataset_type={dataset_type} "
                f"is_partitioned={is_partitioned} "
                f"job_subdir={dataset_type}/{jobid} "
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
