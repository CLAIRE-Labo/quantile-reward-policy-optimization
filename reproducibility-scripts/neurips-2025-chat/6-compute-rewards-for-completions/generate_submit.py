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

job_name = "datasets-with-ref-rewards"

datasets = ["magpieair", "ultrafeedback"]

splits = ["train_split", "eval_split"]

models = ["mistral", "llama"]

temperatures = {
    "mistral": [1],
    "llama": [1],
}

reward_models = ["armorm"]

sftnosfts = ["sft", "nosft"]

dataset_num_ref_rewards = 50

# Takes 30 min in parallel.
# subpartition size = 1024
# partition size = 1024 * 4 = 4096
# nb partitions = 100000 / 4096 <= 25


partition_size = 4096  # 1024 prompts per GPU
save_interval = 1024
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
                                    f"-t 2:00:00 "
                                    f"-N {num_nodes} "
                                    f"-o {stdout_root}/out/{jobid}.out "
                                    f"-e {stdout_root}/out/{jobid}.err "
                                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-compute-rewards-for-completions.sh "
                                    f"reward_model={reward_model} "
                                    f"dataset={dataset_with_chosen_rewards} "
                                    f"dataset_id={dataset_with_ref_completions} "
                                    f"split={split_name} "
                                    f"partition_start_idx={partition_start_idx} "
                                    f"partition_end_idx={partition_end_idx} "
                                    f"save_interval={save_interval} "
                                    "outputs_subdir=shared "
                                    f"job_subdir_prefix={job_name}/{jobid} "
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
