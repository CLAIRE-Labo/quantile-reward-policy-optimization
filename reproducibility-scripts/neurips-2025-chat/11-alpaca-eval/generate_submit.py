from collections import defaultdict
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
job_name = "chat-baseline"

dataset_with_ref_rewards_path_prefix = (
    "\${data_dir}/shared/datasets-with-ref-rewards/merged"
)
model_sftnosft_path_prefix = "\${outputs_dir}/shared/train_sft/sft-chosen/"
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

train_qrpo_path_prefix = "\${outputs_dir}/shared/train_qrpo/"

checkpoints_to_process = [
    model_sftnosft_path_prefix + model_sftnosft_paths["llama-nosft-magpieair-armorm"],
    model_sftnosft_path_prefix
    + model_sftnosft_paths["llama-nosft-ultrafeedback-armorm"],
    model_sftnosft_path_prefix + model_sftnosft_paths["llama-sft-magpieair-armorm"],
    model_sftnosft_path_prefix + model_sftnosft_paths["llama-sft-ultrafeedback-armorm"],
    model_sftnosft_path_prefix + model_sftnosft_paths["mistral-nosft-magpieair-armorm"],
    model_sftnosft_path_prefix
    + model_sftnosft_paths["mistral-nosft-ultrafeedback-armorm"],
    model_sftnosft_path_prefix + model_sftnosft_paths["mistral-sft-magpieair-armorm"],
    model_sftnosft_path_prefix
    + model_sftnosft_paths["mistral-sft-ultrafeedback-armorm"],
    # + All the relevant checkpoints ...
    # DPO, REBEL, qrpo
]
commands = []
eval_commands = defaultdict(list)
nodes_needed = 0
num_nodes = 1

for checkpoint in checkpoints_to_process:
    jobid = "-".join(checkpoint.split("/")[-4:])
    commands.append(
        (
            "sbatch "
            f"-t 1:00:00 "
            f"-N {num_nodes} "
            f"-o {stdout_root}/out/{jobid}.out "
            f"-e {stdout_root}/out/{jobid}.err "
            "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-alpaca.sh "
            "python -m qrpo.evals.run_alpaca_gpt "
            f"checkpoint_path='{checkpoint}' "
            f"job_subdir={job_name}/{jobid} "
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
print(f"Needed {nodes_needed} nodes.")
