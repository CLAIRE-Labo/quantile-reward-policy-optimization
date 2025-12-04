from datetime import datetime
from pathlib import Path

import yaml

"""Nomenclature:

dataset = f"{dataset}"
model = f"{base_model}"
reward_model = f"{reward_model}"

dataset_with_chosen_rewards = f"{dataset}-{reward_model}"

model_sftnosft = f"{model}-(sft|nosft)-{dataset_with_chosen_rewards}"
               = f"{model}-(sft|nosft)-{dataset}-{reward_model}"

dataset_with_ref_completions = f"{model_sftnosft}-(offline|offpolicy|mix)-temp{temperatures}"
                             = f"{model}-(sft|nosft)-{dataset}-{reward_model}-(offline|offpolicy|mix)-temp{temperatures}-ref{NRefDataset}"

dataset_with_ref_rewards = f"{dataset_with_ref_completions}-{reward_model}"
                         = f"{model}-(sft|nosft)-{dataset}-{reward_model}-(offline|offpolicy|mix)-temp{temperatures}-{reward_model}"
"""

stdout_prefix = "all"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "datasets-with-ref-rewards-code"

datasets = ["leetcode"]

splits = ["train_split", "eval_split"]

models = ["llama"]

temperatures = {
    "llama": [1],
}

reward_models = ["sandbox"]

sftnosfts = ["sft", "nosft"]

dataset_num_ref_rewards = 50

# Each node will take prompts with 50 solutions and try to compute the reward for each.
# This is 2600 prompts * 50 completions * 100 tests = 13M code executions.
# we can run completions in parallel using 250 cpus on each node.
# Sequentially run all the tests in the same container for each prompt.
# This takes max 120 seconds.
# 250 completions in parallel let's say, taking 180 seconds total with overhead.
# so 5 prompts in parallel with 50 solutions each, take 180 seconds.

# ~30 minutes per job using 85 jobs.
partition_size = 30
save_interval = 32
num_nodes = 1

commands = []
nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            for temperature in temperatures[model]:
                for sftnosft in sftnosfts:
                    for split in splits:
                        dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                        model_sftnosft = (
                            f"{model}-{sftnosft}-{dataset_with_chosen_rewards}"
                        )
                        dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{dataset_num_ref_rewards}-offline"
                        dataset_with_ref_rewards = (
                            f"{dataset_with_ref_completions}-{reward_model}"
                        )

                        with open(
                            f"src/qrpo/configs/dataset/{dataset_with_chosen_rewards}.yaml",
                            "r",
                        ) as file:
                            dataset_config = yaml.safe_load(file)
                        split_config = dataset_config["dataset_args"][split]
                        split_name = split_config["name"]
                        split_size = split_config["end"] - split_config["start"]

                        for partition_start_idx in range(0, split_size, partition_size):
                            partition_end_idx = min(
                                partition_start_idx + partition_size, split_size
                            )
                            jobid = f"{dataset_with_ref_rewards}/{split}/{partition_start_idx}-{partition_end_idx}"

                            commands.append(
                                (
                                    "sbatch "
                                    f"-t 3:00:00 "
                                    f"-N {num_nodes} "
                                    f"-o {stdout_root}/out/{jobid}.out "
                                    f"-e {stdout_root}/out/{jobid}.err "
                                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-compute-rewards-for-solutions.sh "
                                    f"reward_model={reward_model} "
                                    f"dataset_num_ref_rewards={dataset_num_ref_rewards} "
                                    f"dataset={dataset_with_chosen_rewards} "
                                    f"dataset_id={dataset_with_ref_completions} "
                                    f"split={split_name} "
                                    f"partition_start_idx={partition_start_idx} "
                                    f"partition_end_idx={partition_end_idx} "
                                    f"save_interval={save_interval} "
                                    "outputs_subdir=shared "
                                    f"job_subdir={job_name}/{jobid} "
                                    "resuming.resume=True "
                                )
                            )
                            nodes_needed += num_nodes

# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
