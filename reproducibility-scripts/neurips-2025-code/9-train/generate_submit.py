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
# Also generate the evals
stdout_evals_root = (
    Path(__file__).parent.parent / "10-online-reward-eval"
).resolve().relative_to(
    Path.cwd()
) / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

eval_types = ["eval_split"]

job_name = "code-baseline"

dataset_with_ref_rewards_path_prefix = (
    "\${data_dir}/shared/datasets-with-ref-rewards-code/merged"
)
model_sftnosft_path_prefix = "\${outputs_dir}/shared/train_sft/sft-chosen-code"
model_sftnosft_paths = {
    "llama-nosft-leetcode-sandbox": "llama-nosft-leetcode-sandbox/checkpoints/b72ee3a6bcc20747/",
    "llama-sft-leetcode-sandbox": "llama-sft-leetcode-sandbox/checkpoints/57743cffa2800a0d/checkpoint-60",
}

datasets = ["leetcode"]
reward_models = ["sandbox"]
models = [
    "llama",
]

temperatures = {
    "llama": [1],
}

sftnosfts = [
    "sft",
    "nosft",
]
distributions = [
    "offpolicy10random",
]

dataset_num_ref_rewards = 50
train_num_ref_rewards_list = [20]

losses = [
    "qrpo",
    "dpo",
    "rebel",
    "simpo",
]
betas = {
    "qrpo-identity": [3e-3, 1e-2, 3e-2],
    "dpo-default": [1e-2, 3e-2, 1e-1],
    "rebel-default": [1e-4, 1e-2, 1],
    "simpo-default": [2.0, 2.5, 10.0],
}

qrpo_transform_types = {
    "qrpo": [
        "identity",
    ],
    "dpo": ["default"],
    "rebel": ["default"],
    "simpo": ["default"],
}

qrpo_loss_types = {
    "qrpo": [
        "mse",
    ],
    "dpo": ["default"],
    "rebel": ["default"],
    "simpo": ["default"],
}

qrpo_sample_selectors = {
    "qrpo": [
        "both",
    ],
    "dpo": ["default"],
    "rebel": ["default"],
    "simpo": ["default"],
}

learning_rates = [1e-7, 3e-7, 1e-6]

max_grad_norm = 1e8  # Disable but still log.

batch_size = 128
num_nodes = 2
num_devices_per_node = 4
per_device_train_batch_size = 2
per_device_eval_batch_size = 8
accumulation_steps = batch_size // (
    num_nodes * num_devices_per_node * per_device_train_batch_size
)


commands = []
eval_commands = defaultdict(list)
nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            for temperature in temperatures[model]:
                for sftnosft in sftnosfts:
                    for distribution in distributions:
                        dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                        model_sftnosft = (
                            f"{model}-{sftnosft}-{dataset_with_chosen_rewards}"
                        )
                        dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{dataset_num_ref_rewards}-{distribution}"
                        eval_dataset_with_ref_completions = f"{model_sftnosft}-temp{temperature}-ref{dataset_num_ref_rewards}-offline"
                        dataset_with_ref_rewards = (
                            f"{dataset_with_ref_completions}-{reward_model}"
                        )
                        eval_dataset_with_ref_rewards = (
                            f"{eval_dataset_with_ref_completions}-{reward_model}"
                        )
                        model_sftnosft_path = f"{model_sftnosft_path_prefix}/{model_sftnosft_paths[model_sftnosft]}"
                        dataset_with_ref_rewards_path = f"{dataset_with_ref_rewards_path_prefix}/{dataset_with_ref_rewards}"
                        eval_dataset_with_ref_rewards_path = f"{dataset_with_ref_rewards_path_prefix}/{eval_dataset_with_ref_rewards}"

                        for loss in losses:
                            for loss_transform in qrpo_transform_types[loss]:
                                for qrpo_loss_type in qrpo_loss_types[loss]:
                                    for qrpo_sample_selector in qrpo_sample_selectors[
                                        loss
                                    ]:
                                        for lr in learning_rates:
                                            for beta in betas[
                                                f"{loss}-{loss_transform}"
                                            ]:
                                                for apply_temperature_in_training in [
                                                    False
                                                ]:
                                                    for (
                                                        train_num_ref_rewards
                                                    ) in train_num_ref_rewards_list:
                                                        if (
                                                            apply_temperature_in_training
                                                            and temperature == 1.0
                                                        ):
                                                            continue
                                                        loss_id = (
                                                            f"{loss}-{loss_transform}"
                                                        )
                                                        jobid = f"{dataset_with_ref_rewards}-{loss_id}-applytemp{apply_temperature_in_training}-numref{train_num_ref_rewards}-lr{lr}-beta{beta}"
                                                        commands.append(
                                                            (
                                                                "sbatch "
                                                                f"-t 3:00:00 "
                                                                f"-N {num_nodes} "
                                                                f"-o {stdout_root}/out/{jobid}.out "
                                                                f"-e {stdout_root}/out/{jobid}.err "
                                                                "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                                                                f"-m qrpo.train_qrpo "
                                                                f"dataset={dataset_with_chosen_rewards} "
                                                                f"dataset_args.dataset_path='{dataset_with_ref_rewards_path}' "
                                                                f"model={model} "
                                                                f"model_args.model_name_or_path='{model_sftnosft_path}' "
                                                                f"model_generation_config.temperature={temperature} "
                                                                f"training_args.max_grad_norm={max_grad_norm} "
                                                                f"training_args.gradient_accumulation_steps={accumulation_steps} "
                                                                f"training_args.per_device_train_batch_size={per_device_train_batch_size} "
                                                                f"training_args.per_device_eval_batch_size={per_device_eval_batch_size} "
                                                                f"training_args.learning_rate={lr} "
                                                                f"training_args.loss_type={loss} "
                                                                f"training_args.qrpo_transform_type={loss_transform} "
                                                                f"training_args.qrpo_loss_type={qrpo_loss_type} "
                                                                f"training_args.qrpo_sample_selector={qrpo_sample_selector} "
                                                                f"training_args.apply_temperature_in_training={apply_temperature_in_training} "
                                                                f"training_args.num_ref_rewards={train_num_ref_rewards} "
                                                                f"training_args.beta={beta} "
                                                                f"job_subdir={job_name}/{jobid} "
                                                                f"wandb.run_name={jobid}-{job_name} "
                                                                f"'wandb.tags=[prod,{job_name},{dataset},{reward_model},{model},{sftnosft},{distribution},{loss}]' "
                                                                "outputs_subdir=shared "
                                                                "resuming.resume=True "
                                                            )
                                                        )
                                                        nodes_needed += num_nodes

                                                        for eval_type in eval_types:
                                                            eval_commands[
                                                                eval_type
                                                            ].append(
                                                                (
                                                                    "sbatch "
                                                                    f"-t 3:00:00 "
                                                                    f"-N 1 "
                                                                    f"-o {stdout_evals_root}/{eval_type}/out/{jobid}.out "
                                                                    f"-e {stdout_evals_root}/{eval_type}/out/{jobid}.err "
                                                                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-sandbox.sh "
                                                                    f"python -m qrpo.evals.run_online_eval_code "
                                                                    f"dataset={dataset_with_chosen_rewards} "
                                                                    f"dataset_args.dataset_path='{eval_dataset_with_ref_rewards_path}' "
                                                                    f"model={model} "
                                                                    f"model_args.model_name_or_path='{model_sftnosft_path}' "
                                                                    f"reward_model={reward_model} "
                                                                    "training_dir='\${outputs_dir}/shared/train_qrpo/"
                                                                    f"{job_name}/{jobid}' "
                                                                    f"eval_type={eval_type} "
                                                                    f"training_args.loss_type={loss} "
                                                                    f"training_args.qrpo_transform_type={loss_transform} "
                                                                    f"training_args.qrpo_loss_type={qrpo_loss_type} "
                                                                    f"training_args.qrpo_sample_selector={qrpo_sample_selector} "
                                                                    f"training_args.num_ref_rewards={train_num_ref_rewards} "
                                                                    f"training_args.beta={beta} "
                                                                    f"job_subdir={job_name}/{jobid} "
                                                                    "outputs_subdir=shared "
                                                                    "resuming.resume=True "
                                                                )
                                                            )


# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {nodes_needed//num_nodes} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print(f"Needed {nodes_needed} nodes.")

# Same for eval
for eval_type, commands in eval_commands.items():
    submit_dir = Path.cwd() / str(stdout_evals_root) / eval_type
    submit_dir.mkdir(parents=True, exist_ok=True)
    submit_file = submit_dir / "submit.sh"
    print(f"Writing {len(commands)} commands to {submit_file}")
    with open(submit_file, "w") as f:
        for command in commands:
            f.write(command + "\n")
