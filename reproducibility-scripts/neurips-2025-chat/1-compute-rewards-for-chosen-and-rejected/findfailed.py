#!/usr/bin/env python3

import sys

sys.path.append("..")
from findfailed_utils import find_failed_jobs

success_message = "Rewards computed successfully!"
find_success_n_times = 4
num_resuming_dirs = 4
checkpoint_validator = "state.json"
errors_to_match = [
    "IndexError",
    "FileNotFoundError",
    "Timeout at NCCL work",
    "DUE TO TIME LIMIT",
    "SIGTERM",
    "srun: error",
]

if __name__ == "__main__":
    find_failed_jobs(
        __file__,
        success_message,
        errors_to_match,
        find_success_n_times,
        num_resuming_dirs,
        checkpoint_validator,
    )
