import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import wandb
from accelerate.utils import broadcast_object_list as accelerate_broadcast_object_list
from omegaconf import OmegaConf

from qrpo import utils


def merge_and_save_config(config, script_args, training_args, model_args, acc_state):
    # Full config will be different between processes as it contains the process index.
    full_config = OmegaConf.create(
        {
            "script_args": dataclasses.asdict(script_args),
            "training_args": training_args.to_dict(),
            "model_args": dataclasses.asdict(model_args),
        }
    )
    full_config = OmegaConf.merge(full_config, config)

    utils.config.save_or_check_config(
        full_config,
        f"config/process-{acc_state.process_index}/full-config-trl.yaml",
    )

    return full_config


def setup_config_and_resuming(config, acc_state, acc_logger, postprocess_func=None):
    acc_logger.info(f"Running command: {subprocess.list2cmdline(sys.argv)}")
    acc_logger.info(f"Init directory: {Path.cwd()}")
    if acc_state.is_main_process:
        utils.config.setup_resuming_dir(config)
    config = accelerate_broadcast_object_list([config], from_process=0)[0]
    acc_logger.info(f"Run can be resumed from the directory: {config.resuming_dir}")
    if config.resuming.resume:
        os.chdir(config.resuming_dir)

    Path(f"config/process-{acc_state.process_index}").mkdir(exist_ok=True, parents=True)
    utils.config.save_or_check_config(
        config,
        f"config/process-{acc_state.process_index}/config-raw.yaml",
    )

    # Do some optional postprocessing to the config (e.g., checking division of batch size etc.)
    OmegaConf.resolve(config)
    if postprocess_func:
        config = postprocess_func(config)

    # Save the resolved config.
    utils.config.save_or_check_config(
        config, f"config/process-{acc_state.process_index}/config-postprocessed.yaml"
    )

    acc_logger.info(f"Resuming from the directory: {Path.cwd()}")
    acc_logger.info(f"Running with config: \n{OmegaConf.to_yaml(config)}")
    if config.resuming.resume:
        acc_logger.info(f"Resuming from the directory: {Path.cwd()}")

    return config
