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

eval_types = ["eval_split", "alpaca_eval"]

job_name = "chat-baseline"

dataset_with_ref_rewards_path_prefix = (
    "\${data_dir}/shared/datasets-with-ref-rewards/merged"
)
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

datasets = [
    "magpieair",
    "ultrafeedback",
]
reward_models = [
    "armorm",
]
models = [
    "mistral",
    "llama",
]

temperatures = {
    "mistral": [1],
    "llama": [1],
}

sftnosfts = [
    "sft",
    "nosft",
]
distributions = [
    "offline",
    "offpolicy2best",
    "offpolicy2random",
]

dataset_num_ref_rewards = 50
train_num_ref_rewards_list = [1, 3]

losses = [
    "qrpo",
    "dpo",
    "rebel",
    "simpo",
]
betas = {
    "qrpo-identity": [3e-4, 1e-3, 3e-3],
    # "qrpo-log": [3e-4, 1e-3, 3e-3],
    # "qrpo-normal-icdf": [3e-2, 1e-1, 3e-1],
    # "qrpo-scaled-normal-icdf": [3e-2, 1e-1, 3e-1],
    "dpo-default": [1e-2, 3e-2, 1e-1],  # like the SimPO paper did, quite common.
    "rebel-default": [1e-6, 1e-4, 1e-2],  # Same as their ultrafeedback + llama.
    "simpo-default": [
        2.0,
        2.5,
        10,
    ],  # Same as the SimPO paper and what they recommend on GitHub.
}

qrpo_transform_types = {
    "qrpo": [
        # "log",
        "identity",
        # "normal-icdf",
        # "scaled-normal-icdf",
    ],
    "dpo": ["default"],
    "rebel": ["default"],
    "simpo": ["default"],
}

qrpo_loss_types = {
    "qrpo": [
        "mse",
        # "mae",
    ],
    "dpo": ["default"],
    "rebel": ["default"],
    "simpo": ["default"],
}

qrpo_sample_selectors = {
    "qrpo": [
        "both",
        # "chosen",
        # "rejected",
        # "random",
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
                        dataset_with_ref_rewards = (
                            f"{dataset_with_ref_completions}-{reward_model}"
                        )
                        model_sftnosft_path = f"{model_sftnosft_path_prefix}/{model_sftnosft_paths[model_sftnosft]}"
                        dataset_with_ref_rewards_path = f"{dataset_with_ref_rewards_path_prefix}/{dataset_with_ref_rewards}"

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
                                                        if (
                                                            train_num_ref_rewards > 1
                                                            and loss != "qrpo"
                                                        ):
                                                            # Not used by dpo/rebel/simpo.
                                                            continue
                                                        if (
                                                            train_num_ref_rewards == 1
                                                            and loss == "qrpo"
                                                            and dataset != "magpieair"
                                                        ):
                                                            # num_ref = 1 is a special showoff for magpieair only.
                                                            # Ultrafeedback should use 3.
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
                                                                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended.sh "
                                                                    f"python -m qrpo.evals.run_online_eval "
                                                                    f"dataset={dataset_with_chosen_rewards} "
                                                                    f"dataset_args.dataset_path='{dataset_with_ref_rewards_path}' "
                                                                    f"model={model} "
                                                                    f"model_args.model_name_or_path='{model_sftnosft_path}' "
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
