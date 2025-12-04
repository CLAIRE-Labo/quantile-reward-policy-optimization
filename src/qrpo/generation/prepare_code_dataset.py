import logging
import re

import datasets
import hydra
import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer

from qrpo import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)

pre_code = """import collections
import itertools
import functools
import math
import string
import random
import bisect
import re
import operator
import heapq
import queue

from typing import List, Tuple, Dict, Any, Union, Optional
from queue import PriorityQueue
from itertools import combinations, permutations
from functools import lru_cache
from collections import defaultdict
from collections import OrderedDict
from collections import deque
from collections import Counter

inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

class Node:
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class CategoryHandler:
    def haveSameCategory(self, a: int, b: int) -> bool:
        pass
"""


def add_import_prompt(prompt: str) -> str:
    # Reformat format
    prompt = re.sub(r"### Format:\s+", "### Format:\n", prompt, count=1)
    # Add mention of imports to starter code:
    prompt = re.sub(
        r"```python\s*\n",
        "```python\n# Additional imports.\n\n# End of additional imports.\n\n",
        prompt,
        count=1,
    )
    prompt = re.sub(
        r"(### Format:\n.*?\n)",  # after the original description line
        r"\1You have to add the missing imports to the starter code to run correctly.\n",
        prompt,
        count=1,
        flags=re.DOTALL,
    )
    # Add the pre_code to the prompt.
    prompt = re.sub(
        r"(### Format:\s+)",  # after the original description line
        rf"\1Your code will run after the following definitions and imports, which may or may not be needed.\n```python\n{pre_code}```\n\n",
        prompt,
        count=1,
        flags=re.DOTALL,
    )

    return prompt


def preprocess(example, tokenizer):
    """Construct the chosen and rejected columns and add
    chosen_reward_tokens_len (num in characters).
    rejected_reward_tokens_len (num in characters).
    max_chosen_rejected_reward_tokens_len (num in characters).
    """
    # Change the query to ask the model to add the necessary imports.
    prompt = example["query"]
    prompt = add_import_prompt(prompt)

    # chosen is the chat: query + response
    # rejected is same as chosen. We don't have a rejected response.
    # chosen and rejected_rewards should be 1. No verification.
    example["chosen"] = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]
    example["rejected"] = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]

    # tokenize.

    tokenized_message = tokenizer.apply_chat_template(example["chosen"])
    message_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
    chosen_reward_tokens_len = len(tokenized_message)
    example["chosen_reward_tokens"] = tokenized_message
    example["rejected_reward_tokens"] = tokenized_message
    example["chosen_reward_tokens_len"] = chosen_reward_tokens_len
    example["rejected_reward_tokens_len"] = chosen_reward_tokens_len
    example["max_chosen_rejected_reward_tokens_len"] = chosen_reward_tokens_len
    example["chosen_reward_text"] = message_text
    example["rejected_reward_text"] = message_text
    example["chosen_rewards"] = 1
    example["rejected_rewards"] = 1

    # Tests
    example["num_tests"] = len(example["input_output"])

    return example


def filter_problem(example, config):
    # Filter problems with input files
    min_tests = config.reward_model_args.min_tests
    if example["num_tests"] < min_tests:
        return False
    if example["max_chosen_rejected_reward_tokens_len"] > config.max_seq_length:
        return False
    return True


@hydra.main(
    version_base=None, config_path="../configs", config_name="prepare-code-dataset"
)
def main(config: DictConfig) -> None:
    """
    Convert the dataset to a preference dataset.
    """
    config = utils.config.setup_config_and_resuming(config)
    ds = datasets.load_from_disk(config.dataset_args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_args.model_name_or_path
    )
    ds = ds.map(preprocess, fn_kwargs={"tokenizer": tokenizer})
    print(ds["train"][0]["chosen"][0]["content"])
    print(ds["train"][100]["chosen"][0]["content"])
    print(ds)
    for split in ["train", "test"]:
        l = ds[split][:]["max_chosen_rejected_reward_tokens_len"]
        mean_lengths = np.array(l).mean()
        max_lengths = np.array(l).max()
        print(f"Mean length of {split} set: {mean_lengths}")
        print(f"Max length of {split} set: {max_lengths}")
        # distribution of difficulty ds[split][:]["difficulty"]
        d = ds[split][:]["difficulty"]
        print("Easy: ", len([x for x in d if x == "Easy"]) / len(d))
        print("Medium: ", len([x for x in d if x == "Medium"]) / len(d))
        print("Hard: ", len([x for x in d if x == "Hard"]) / len(d))

    ds = ds.filter(filter_problem, fn_kwargs={"config": config})
    print(ds)
    for split in ["train", "test"]:
        l = ds[split][:]["max_chosen_rejected_reward_tokens_len"]
        mean_lengths = np.array(l).mean()
        max_lengths = np.array(l).max()
        print(f"Mean length of {split} set: {mean_lengths}")
        print(f"Max length of {split} set: {max_lengths}")
        d = ds[split][:]["difficulty"]
        print("Easy: ", len([x for x in d if x == "Easy"]) / len(d))
        print("Medium: ", len([x for x in d if x == "Medium"]) / len(d))
        print("Hard: ", len([x for x in d if x == "Hard"]) / len(d))

    logger.info(f"Saving merged dataset to {config.save_path}")
    ds.save_to_disk(config.save_path)
    logger.info("Dataset saved successfully.")


if __name__ == "__main__":
    main()
