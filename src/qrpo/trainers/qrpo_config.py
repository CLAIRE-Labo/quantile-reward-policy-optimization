# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Literal, Optional

from transformers import TrainingArguments

from qrpo.trainers.trl_dpo_config import FDivergenceType


@dataclass
class QRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`QRPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model.
        simpo_gamma_beta_ratio (`float`, *optional*, defaults to `0.1`):
            Ratio between the beta and gamma parameters in the SimPo loss.
        num_ref_rewards (`int`, *optional*, defaults to `None`):
            Number of reference rewards to use when computing quantiles.
        loss_type (`str`, *optional*, defaults to `"qrpo"`):
            Type of loss to use. Possible values are: `"qrpo"`, `"dpo"`, `"rebel"`, `"simpo"`.
        qrpo_transform_type (`str`, *optional*, defaults to `"log"`):
            Type of transformation to use in QRPO loss (applied only if `loss_type="qrpo"`). Possible values are: `"log"`, `"identity"`, `"normal-icdf"`, `"scaled-normal-icdf"`.
        qrpo_loss_type (`str`, *optional*, defaults to `"mse"`):
            Type of loss to use in QRPO loss (applied only if `loss_type="qrpo"`). Possible values are: `"mse"`, `"mae"`.
        qrpo_sample_selector (`str`, *optional*, defaults to `"both"`):
            Type of sample selector to use in QRPO loss (applied only if `loss_type="qrpo"`). Possible values are:
            `"both"`, `"chosen"`, `"rejected"`, `"random"`.
            - `"both"`: both chosen and rejected samples are used.
            - `"chosen"`: only chosen samples are used.
            - `"rejected"`: only rejected samples are used.
            - `"random"`: randomly choose from chosen and rejected samples.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature to use for log-probs computation during training and generation
        apply_temperature_in_training (`bool`, *optional*, defaults to `False`):
            Whether to apply temperature scaling to the logits during training, this is useful for QRPO so that the
            reference model is faithful to the reference rewards used to compute the quantiles.
            (temperature is always applied during generation.)
        top_p (`float`, *optional*, defaults to `1.0`):
            top_p to use for log-probs computation during training and generation
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        pad_token_id (`Optional[int]`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the
            default data collator.
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        max_prompt_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the target. This argument is required if you want to use the default data collator and
            your model is an encoder-decoder.
        reward_model_max_input_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch that is passed to the reward model.
        is_encoder_decoder(`Optional[int]`, *optional*, defaults to `None`):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            If `True`, generates and logs completions from both the model and the reference model to W&B during
            evaluation.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute reference model log probabilities for training and evaluation datasets. This is
            useful when training without the reference model to reduce the total GPU memory needed.
        precompute_ref_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Batch size to use when precomputing reference model log probabilities. This can be set higher than the
            training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
            training and `per_device_eval_batch_size` for evaluation.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        model_init_kwargs (`Optional[dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        ref_model_init_kwargs (`Optional[dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model
            from a string.
        model_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`, *optional*, defaults to `False`):
            If `True`, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal
            probability to all responses.
        force_use_ref_model (`bool`, *optional*, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the
            ref_model, set this flag to `True`.
        f_divergence_type (`str`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            Type of f-divergence regularization function to compute divergence between policy and reference model.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            α coefficient in the α-divergence u^-α regularization function for DPO loss.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            When set to `True`, the reference model is synchronized with the active model every `ref_model_sync_steps`
            steps, using the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`
            To use this parameter, you must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.
        use_num_logits_to_keep (`bool`, *optional*, defaults to `False`):
            If `True`, only a specified number of logits are computed in the forward pass of CausalLM. This can be useful
            for saving memory and speeding up training by not computing the logits for all tokens, especially in scenarios
            when working with very long prompts where labels are -ignored (-100).
            [Read more](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaForCausalLM)
    """

    learning_rate: float = 1e-6
    beta: float = 0.1
    num_ref_rewards: Optional[int] = None
    reward_model_per_device_batch_size: Optional[int] = None
    reward_model_per_device_eval_batch_size: Optional[
        int
    ] = None  # Note: not used for now.
    loss_type: Literal["qrpo", "dpo", "rebel", "simpo"] = "qrpo"
    qrpo_transform_type: Literal[
        "identity",
        "log",
        "normal-icdf",
        "scaled-normal-icdf",
    ] = "log"
    qrpo_loss_type: Literal["mse", "mae"] = "mse"
    qrpo_sample_selector: Optional[
        Literal["both", "chosen", "rejected", "random"]
    ] = "both"
    simpo_gamma_beta_ratio: float = 0.5
    temperature: float = 1.0
    apply_temperature_in_training: bool = False
    top_p: float = 1.0
    label_pad_token_id: int = -100
    pad_token_id: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    reward_model_max_input_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    precompute_ref_batch_size: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[dict[str, Any]] = None
    ref_model_init_kwargs: Optional[dict[str, Any]] = None
    reward_model_init_kwargs: Optional[dict[str, Any]] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: FDivergenceType = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: float = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    use_num_logits_to_keep: bool = False

    def __post_init__(self):
        return super().__post_init__()
