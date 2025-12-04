from pathlib import Path

import numpy as np
from datasets import load_from_disk
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

SELECTION = "selection"
REPORTING = "reporting"


def get_model_data(model, dataset):
    if model == "llama":
        if dataset == "magpieair":
            data_QRPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-sft-magpieair-armorm-temp1-ref50-offline-armorm-qrpo-identity-applytempFalse-numref1-lr1e-06-beta0.0003/checkpoints/ec571a0412863c3f/checkpoint-160/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_DPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-sft-magpieair-armorm-temp1-ref50-offline-armorm-dpo-default-applytempFalse-numref6-lr1e-06-beta0.01/checkpoints/7b2833fa582785d6/checkpoint-160/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_REBEL = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-sft-magpieair-armorm-temp1-ref50-offline-armorm-rebel-default-applytempFalse-numref6-lr3e-07-beta1e-06/checkpoints/e9c9764ff3eb6623/checkpoint-764/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_SIMPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-sft-magpieair-armorm-temp1-ref50-offline-armorm-simpo-default-applytempFalse-numref6-lr1e-06-beta10.0/checkpoints/25bd2c38ddce2e3d/checkpoint-160/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
        elif dataset == "ultrafeedback":
            data_QRPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-sft-ultrafeedback-armorm-temp1-ref50-offline-armorm-qrpo-identity-applytempFalse-numref3-lr3e-07-beta0.001/checkpoints/c57e2819be9c13c6/checkpoint-100/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_DPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm-dpo-default-applytempFalse-numref6-lr3e-07-beta0.01/checkpoints/2c429a37b5514750/checkpoint-300/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_REBEL = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm-rebel-default-applytempFalse-numref6-lr1e-06-beta1e-06/checkpoints/599a6c653ad6e9af/checkpoint-100/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_SIMPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm-simpo-default-applytempFalse-numref6-lr1e-06-beta2.0/checkpoints/722b660f4d92aa69/checkpoint-200/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
        else:
            raise ValueError("Invalid dataset name")

    elif model == "mistral":
        if dataset == "magpieair":
            data_QRPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-sft-magpieair-armorm-temp1-ref50-offline-armorm-qrpo-identity-applytempFalse-numref1-lr3e-07-beta0.0003/checkpoints/725a23754739ecd0/checkpoint-320/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_DPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-sft-magpieair-armorm-temp1-ref50-offline-armorm-dpo-default-applytempFalse-numref6-lr3e-07-beta0.03/checkpoints/9a9c9919922a2aa3/checkpoint-320/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_REBEL = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-sft-magpieair-armorm-temp1-ref50-offline-armorm-rebel-default-applytempFalse-numref6-lr3e-07-beta0.0001/checkpoints/054713ba984f141c/checkpoint-320/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_SIMPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm-simpo-default-applytempFalse-numref6-lr3e-07-beta10.0/checkpoints/8999b25ff0c8b12e/checkpoint-160/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
        elif dataset == "ultrafeedback":
            data_QRPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm-qrpo-identity-applytempFalse-numref3-lr1e-06-beta0.003/checkpoints/8722d3bc88cfc087/checkpoint-200/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_DPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm-dpo-default-applytempFalse-numref6-lr1e-06-beta0.03/checkpoints/0ab7977fca7a295f/checkpoint-100/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_REBEL = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-sft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm-rebel-default-applytempFalse-numref6-lr1e-06-beta0.0001/checkpoints/adb7b9dcfce179ba/checkpoint-100/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
            data_SIMPO = load_from_disk(
                f"outputs/shared/train_qrpo/chat-baseline/mistral-sft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm-simpo-default-applytempFalse-numref6-lr3e-07-beta10.0/checkpoints/4fe965e356efd5fc/checkpoint-200/online_evals_0/eval_split/modified-and-lc-rewards-v10-dataset-{REPORTING}"
            )
        else:
            raise ValueError("Invalid dataset name")

    else:
        raise ValueError("Invalid model name")

    return data_QRPO, data_DPO, data_REBEL, data_SIMPO


# ---------------------------------------------------------------------------
# Global style tweaks (can be overridden locally)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")  # clean white background w/ subtle grid
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.frameon": True,
        "legend.fontsize": 14,
    }
)

# ---------------------------------------------------------------------------
# Visual identity for each algorithm
# ---------------------------------------------------------------------------
MARKERS = {
    "DPO": "s",  # square
    "SimPO": "D",  # diamond
    "REBEL": "^",  # circle
    "QRPO": "o",  # triangle‑up
}

COLORS = {
    "DPO": "cornflowerblue",
    "SimPO": "lightskyblue",
    "REBEL": "mediumseagreen",
    "QRPO": "darkorange",
}

# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------


def plot_trend_for_completion_length_vs_implicit_reward_window(
    data,
    label,
    color="tab:blue",
    marker="o",
    normalize_y=False,
    *,
    scatter_size=10,
    scatter_alpha=0.8,
    linewidth=3.5,
    center_size=90,
):
    completion_length = np.asarray(data["reward_tokens_len"])
    implicit_reward = np.asarray(data["reverse/kl1_seqs"])
    lc_ref_l_std = np.asarray(data["lc_ref_l_std"])
    lc_ref_l_mean = np.asarray(data["lc_ref_l_mean"])

    # remove points where std data["lc_ref_l_std"] == 0
    mask = lc_ref_l_std != 0
    lc_ref_l_std = lc_ref_l_std[mask]
    lc_ref_l_mean = lc_ref_l_mean[mask]
    completion_length = completion_length[mask]
    implicit_reward = implicit_reward[mask]

    x = (completion_length - lc_ref_l_mean) / lc_ref_l_std
    if normalize_y:
        y = (implicit_reward - np.mean(implicit_reward)) / np.std(implicit_reward)
    else:
        y = implicit_reward

    # ------------------------------------------------------------------
    # 1. Remove extreme outliers
    # ------------------------------------------------------------------

    # # --- light trimming of extreme outliers ----------------------------
    x_ci_min, x_ci_max = np.percentile(x, [3, 97])
    y_ci_min, y_ci_max = np.percentile(y, [3, 97])
    mask = np.ones_like(x, dtype=bool)
    mask &= (x > -2) & (x < 5)
    mask &= (x > x_ci_min) & (x < x_ci_max)
    mask &= (y > y_ci_min) & (y < y_ci_max)
    x, y = x[mask], y[mask]

    # -------------------------------------------------------
    # 2. Pplot the data
    # -------------------------------------------------------

    slope, intercept = np.polyfit(x, y, 1)
    y_line = slope * x + intercept
    corr, _ = spearmanr(x, y)
    plt.scatter(
        x,
        y,
        color=color,
        s=scatter_size,
        alpha=scatter_alpha,
        marker=marker,
        edgecolors="white",
        linewidths=0.3,
    )
    # linear trend line
    plt.plot(x, y_line, color=color, linestyle="solid", linewidth=linewidth)

    x_center = x.mean()
    y_x_center = slope * x_center + intercept
    # Plot big marker at the center of the data
    plt.scatter(
        x_center,
        y_x_center,
        color=color,
        s=center_size,
        marker=marker,
        edgecolors="#222222",
        linewidths=0.4,
        label=rf"{label}, $\rho={corr:.2f}$",
        zorder=5,
    )

    plt.xlabel("Completion length (normalized)")
    plt.ylabel("Implicit reward (normalized)")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for model in ["llama", "mistral"]:
        for dataset in ["magpieair", "ultrafeedback"]:
            plt.figure(figsize=(8, 6))
            data_QRPO, data_DPO, data_REBEL, data_SIMPO = get_model_data(model, dataset)
            for data, algo in zip(
                [data_DPO, data_SIMPO, data_REBEL, data_QRPO],
                ["DPO", "SimPO", "REBEL", "QRPO"],
            ):
                plot_trend_for_completion_length_vs_implicit_reward_window(
                    data,
                    label=algo,
                    color=COLORS[algo],
                    marker=MARKERS[algo],
                )

            plt.legend()
            out_path = Path(__file__).parent / f"length-bias"
            out_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path / f"length-bias-{model}-{dataset}.png", dpi=300)
            plt.show()
