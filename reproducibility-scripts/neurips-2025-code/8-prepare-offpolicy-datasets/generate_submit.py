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

datasets = ["leetcode"]

models = ["llama"]

temperatures = {
    "llama": [1],
}

reward_models = ["sandbox"]

sftnosfts = ["sft", "nosft"]
modes = ["offpolicy10random"]

dataset_num_ref_rewards = 50

num_nodes = 1
commands = []
nodes_needed = 0
for dataset in datasets:
    for model in models:
        for temperature in temperatures[model]:
            for sftnosft in sftnosfts:
                for reward_model in reward_models:
                    for mode in modes:
                        dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                        model_sftnosft = (
                            f"{model}-{sftnosft}-{dataset_with_chosen_rewards}"
                        )
                        dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{dataset_num_ref_rewards}-offline"
                        new_dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{dataset_num_ref_rewards}-{mode}"
                        dataset_with_ref_rewards = (
                            f"{dataset_with_ref_completions}-{reward_model}"
                        )
                        new_dataset_with_ref_rewards = (
                            f"{new_dataset_with_ref_completions}-{reward_model}"
                        )
                        jobid = new_dataset_with_ref_rewards
                        commands.append(
                            (
                                "sbatch "
                                f"-t 30:00 "
                                f"-N {num_nodes} "
                                f"-o {stdout_root}/out/{jobid}.out "
                                f"-e {stdout_root}/out/{jobid}.err "
                                "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended.sh "
                                f"python -m qrpo.generation.prepare_offpolicy_dataset "
                                f"dataset={dataset_with_chosen_rewards} "
                                f"dataset_id={dataset_with_ref_rewards} "
                                f"new_dataset_id={new_dataset_with_ref_rewards} "
                                f"mode={mode} "
                                f"exp=-code "
                            )
                        )
                        nodes_needed += num_nodes

# Create output directory
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
