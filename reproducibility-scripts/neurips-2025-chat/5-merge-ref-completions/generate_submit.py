from datetime import datetime
from pathlib import Path

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

datasets = ["magpieair", "ultrafeedback"]

models = ["mistral", "llama"]

temperatures = {
    "mistral": [1],
    "llama": [1],
}

reward_models = ["armorm"]

sftnosfts = ["sft", "nosft"]

dataset_num_ref_reward = 50

is_partitioned = True
dataset_type = "datasets-with-ref-completions"
num_nodes = 1


commands = []
nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            for temperature in temperatures[model]:
                for sftnosft in sftnosfts:
                    dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                    model_sftnosft = f"{model}-{sftnosft}-{dataset_with_chosen_rewards}"
                    dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{dataset_num_ref_reward}-offline"
                    jobid = dataset_with_ref_completions
                    commands.append(
                        (
                            "sbatch "
                            f"-N {num_nodes} "
                            f"-o {stdout_root}/out/{jobid}.out "
                            f"-e {stdout_root}/out/{jobid}.err "
                            "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended.sh "
                            f"python -m qrpo.generation.merge_partitions "
                            f"dataset={dataset_with_chosen_rewards} "
                            f"dataset_id={dataset_with_ref_completions} "
                            f"dataset_type={dataset_type} "
                            f"is_partitioned={is_partitioned} "
                            f"job_subdir={dataset_type}/{jobid} "
                            "outputs_subdir=shared "
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
