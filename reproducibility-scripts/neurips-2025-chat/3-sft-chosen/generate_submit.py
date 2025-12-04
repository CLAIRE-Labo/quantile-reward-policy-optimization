from datetime import datetime
from pathlib import Path

"""Nomenclature:

dataset = f"{dataset}"
model = f"{base_model}"
reward_model = f"{reward_model}"

dataset_with_chosen_rewards = f"{dataset}-{reward_model}"

model_sftnosft = f"{model}-(sft|nosft)-{dataset_with_chosen_rewards}"
               = f"{model}-(sft|nosft)-{dataset}-{reward_model}"
"""

stdout_prefix = "all"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "sft-chosen"

datasets = ["magpieair", "ultrafeedback"]

models = ["mistral", "llama"]

reward_models = ["armorm"]

sftnosfts = ["sft", "nosft"]

batch_size = 128

num_nodes = 1
num_devices_per_node = 4
per_device_train_batch_size = 2
accumulation_steps = batch_size // (
    num_nodes * num_devices_per_node * per_device_train_batch_size
)
per_device_eval_batch_size = 2

max_grad_norm = 10.0
learning_rate = 5e-7

commands = []
nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            for sftnosft in sftnosfts:
                dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                model_sftnosft = f"{model}-{sftnosft}-{dataset_with_chosen_rewards}"
                jobid = model_sftnosft

                num_epochs = 1 if sftnosft == "sft" else 0
                commands.append(
                    (
                        "sbatch "
                        f"-N {num_nodes} "
                        f"-o {stdout_root}/out/{jobid}.out "
                        f"-e {stdout_root}/out/{jobid}.err "
                        "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                        "-m qrpo.train_sft "
                        f"training_args.gradient_accumulation_steps={accumulation_steps} "
                        f"training_args.per_device_train_batch_size={per_device_train_batch_size} "
                        f"training_args.per_device_eval_batch_size={per_device_eval_batch_size} "
                        f"model={model} "
                        f"dataset={dataset_with_chosen_rewards} "
                        f"training_args.num_train_epochs={num_epochs} "
                        f"training_args.max_grad_norm={max_grad_norm} "
                        f"training_args.learning_rate={learning_rate} "
                        f"job_subdir={job_name}/{jobid} "
                        f"wandb.run_name={jobid}-{job_name} "
                        f"'wandb.tags=[prod,{job_name},{dataset},{reward_model},{model},{sftnosft},base-model]' "
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
print(f"Needed {nodes_needed} nodes.")
