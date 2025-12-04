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

job_name = "datasets-with-ref-completions"

datasets = ["magpieair", "ultrafeedback"]

splits = ["train_split", "eval_split"]

models = ["mistral", "llama"]

temperatures = {
    "mistral": [1],
    "llama": [1],
}

reward_models = ["armorm"]

sftnosfts = ["sft", "nosft"]

model_sftnosft_path_prefix = "\${outputs_dir}/shared/train_sft/sft-chosen"
model_sftnosft_paths = {
    "llama-nosft-magpieair-armorm": "llama-nosft-magpieair-armorm/checkpoints/8a2b6e8b7e8f33d2",
    "llama-sft-magpieair-armorm": "llama-sft-magpieair-armorm/checkpoints/d8ef3e6e43845778/checkpoint-764",
    "llama-nosft-ultrafeedback-armorm": "llama-nosft-ultrafeedback-armorm/checkpoints/6fca9a01cec5f5f8",
    "llama-sft-ultrafeedback-armorm": "llama-sft-ultrafeedback-armorm/checkpoints/3981f37e85e943cc/checkpoint-476",
    "mistral-nosft-magpieair-armorm": "mistral-nosft-magpieair-armorm/checkpoints/00869cd7a5e1cc80",
    "mistral-sft-magpieair-armorm": "mistral-sft-magpieair-armorm/checkpoints/4b1b28c59a7d96f6/checkpoint-764",
    "mistral-nosft-ultrafeedback-armorm": "mistral-nosft-ultrafeedback-armorm/checkpoints/60c56c56f11cf12e",
    "mistral-sft-ultrafeedback-armorm": "mistral-sft-ultrafeedback-armorm/checkpoints/d77603a98aef7c13/checkpoint-476",
}

n_completions = 50

# Reference numbers:
## ~6000 tokens per second
# 128 prompts with 100 completions each take 15 minutes on 1 GPU
# 1024 prompts with 50 completions each take 1h on 1 GPU

# We want
# 100k prompts

# completions
# subpartition size = 1024
# partition size = 1024 * 4 = 4096
# nb partitions = 100000 / 4096 <= 25

partition_size = 4096  # 4 GPUs per node, 1024 prompts per GPU
save_interval = 1024  # 1h per GPU
num_nodes = 1  # 1 node per job

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
