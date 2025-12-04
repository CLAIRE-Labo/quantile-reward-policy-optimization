from typing import Any

import requests
from omegaconf import DictConfig, OmegaConf, omegaconf


class SandboxClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def test_connection(self):
        return requests.get(f"{self.url}/test").json()

    def execute_batch_code(
        self,
        pre_code: str,
        batch_ref_model_completions: list[list[str]],  # [B, Nref, str]
        batch_entrypoints: list[str],  # [B, str]
        batch_tests: list[
            str
        ],  # [B, str]    # Each str is the list of assertions to test.
        config: DictConfig,
    ) -> Any:
        res = requests.post(
            f"{self.url}/execute_batch_code/",
            json={
                "pre_code": pre_code,
                "batch_ref_model_completions": batch_ref_model_completions,
                "batch_entrypoints": batch_entrypoints,
                "batch_tests": batch_tests,
                "config": OmegaConf.to_container(config),
            },
        )
        if res.status_code != 200:
            raise Exception(f"Error: {res.status_code} {res.json()['detail']}")
        else:
            return res.json()["results"]
