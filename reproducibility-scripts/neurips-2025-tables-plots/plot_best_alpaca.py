import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SELECTION = "selection"
REPORTING = "reporting"

parsed_results_file = "neurips-2025-chat-baseline-20250718-202520.csv"
out_path_csv = Path("outputs/shared/parsed-results") / f"{parsed_results_file}"
all_df = pd.read_csv(out_path_csv)

# Quickly take the best models.
dataset = "magpieair"
model = "llama"
regime = "offline"
sftnosft = "sft"
mask = all_df["wandb.run_name"].str.match(
    f"{model}-{sftnosft}-{dataset}-.*-{regime}-.*"
)
filtered = all_df.loc[mask]

# collapsed reward-hacking safeguard (solid heuristic)
keep = (
    filtered[f"online_eval/eval_split/{SELECTION}/avg_reward_tokens_len_mean"] <= 1000
)
keep = (
    keep & filtered[f"online_eval/eval_split/{REPORTING}/avg_reward_tokens_len_mean"]
    <= 1000
)
filtered = filtered.loc[keep]

# Pick best checkpoint.
# 12 fields to include all relevant info apart from LR and beta.
filtered["to_pick_from"] = (
    filtered["wandb.run_name"].str.split("-").str[:12].str.join("-")
)
best_indices = filtered.groupby("to_pick_from")[
    f"online_eval/eval_split/{SELECTION}/avg_reward_mean"
].idxmax()
best_models = filtered.loc[best_indices]
best_models = best_models.sort_values(
    "alpaca/length_controlled_winrate", ascending=True
)
names = best_models["training_args.loss_type"].values
scores = best_models["alpaca/length_controlled_winrate"].values
stds = best_models["alpaca/lc_standard_error"].values
labels = [
    {
        "rebel": "REBEL",
        "dpo": "DPO",
        "simpo": "SimPO",
        "qrpo": r"$\bf{QRPO\ (Ours)}$",
    }[name]
    for name in names
]
colors = [
    {
        "rebel": "mediumseagreen",
        "dpo": "cornflowerblue",
        "simpo": "lightskyblue",
        "qrpo": "darkorange",
    }[name]
    for name in names
]

alpha = 0.85

# only left plot
fig, ax = plt.subplots(ncols=1, figsize=(6.5, 4.5))
plt.subplots_adjust(bottom=0.8)


# ---- Bar Plot ----
x = np.arange(len(labels))
bar_width = 0.6

# LC Score bars
ax.bar(
    x,
    scores,
    yerr=stds,
    capsize=5,
    color=colors,
    edgecolor="black",
    width=bar_width,
    alpha=alpha,
    label="LC Score",
)

# Annotations
for xi, score, name in zip(x, scores, names):
    fontweight = "bold" if name == "qrpo" else "normal"
    ax.text(
        xi,
        score + 0.3,
        f"{score:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight=fontweight,
    )

# Style
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.set_title("Best AlpacaEval 2 LC Scores (↑)", fontsize=17)
ax.set_ylabel("LC Score", fontsize=14)
ax.set_ylim(44, 51.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.axhline(50, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
# add a label for the line
ax.text(
    x=0.7,  # or any x‐position that looks good
    y=50 + 0.15,  # slight vertical offset above the line
    s="gpt-4-1106-preview",
    color="red",
    alpha=0.6,
    fontsize=11,
    fontstyle="italic",
    ha="left",
    va="bottom",
)

plt.tight_layout()
out_path = Path(__file__).parent / "figure-1"
out_path.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path / "best-models-alpaca.png", dpi=300)

plt.show()
