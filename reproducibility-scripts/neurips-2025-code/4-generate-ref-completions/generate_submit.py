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

job_name = "datasets-with-ref-completions-code"

datasets = ["leetcode"]

splits = ["train_split", "eval_split"]

models = ["llama"]

temperatures = {
    "llama": [1],
}

reward_models = ["sandbox"]

sftnosfts = ["sft", "nosft"]

model_sftnosft_path_prefix = "\${outputs_dir}/shared/train_sft/sft-chosen-code"
model_sftnosft_paths = {
    "llama-nosft-leetcode-sandbox": "llama-nosft-leetcode-sandbox/checkpoints/b72ee3a6bcc20747/",
    "llama-sft-leetcode-sandbox": "llama-sft-leetcode-sandbox/checkpoints/57743cffa2800a0d/checkpoint-60",
}

n_completions = 50

partition_size = 2600  # 660 prompts per GPU
save_interval = 1024  # 1h per GPU
num_nodes = 1  # 1 GPU per node

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
                        dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{n_completions}-offline"

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
                            jobid = f"{dataset_with_ref_completions}/{split}/{partition_start_idx}-{partition_end_idx}"
                            model_path = f"{model_sftnosft_path_prefix}/{model_sftnosft_paths[model_sftnosft]}"

                            commands.append(
                                (
                                    "sbatch "
                                    "-t 1:30:00 "
                                    f"-N {num_nodes} "
                                    f"-o {stdout_root}/out/{jobid}.out "
                                    f"-e {stdout_root}/out/{jobid}.err "
                                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-generate-ref-completions-with-vllm.sh "
                                    f"model={model} "
                                    f"model_args.model_name_or_path='{model_path}' "
                                    f"model_generation_config.temperature={temperature} "
                                    f"dataset={dataset_with_chosen_rewards} "
                                    f"split={split_name} "
                                    f"n_completions={n_completions} "
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
