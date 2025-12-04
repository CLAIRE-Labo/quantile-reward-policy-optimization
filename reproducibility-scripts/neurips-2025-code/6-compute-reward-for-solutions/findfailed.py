#!/usr/bin/env python3

import sys

sys.path.append("..")
from findfailed_utils import find_failed_jobs

success_message = "Rewards computed successfully!"
already_done_message = (
    "All samples in the subpartition have already been processed. Exiting."
)
find_success_n_times = 1
num_resuming_dirs = 1
checkpoint_validator = "state.json"

if __name__ == "__main__":
    find_failed_jobs(
        __file__,
        success_message,
        find_success_n_times,
        num_resuming_dirs,
        checkpoint_validator,
        already_done_message,
    )
