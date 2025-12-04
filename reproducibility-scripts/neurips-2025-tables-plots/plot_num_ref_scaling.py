from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SELECTION = "selection"
REPORTING = "reporting"


def plot_scaling(df, lr_name):
    plt.style.use("seaborn-v0_8-whitegrid")  # clean white background w/ subtle grid
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 32,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.frameon": True,
            "legend.fontsize": 24,
            "lines.linewidth": 3.5,
        }
    )
    colors_dicr = {
        "offline": {
            0.003: "#006D63",
            0.01: "#006D63",
            0.1: "#006D63",
        },  # teal family
        "offpolicy2random": {
            0.003: "#9B0E4C",
            0.01: "#9B0E4C",
            0.1: "#9B0E4C",
        },  # magenta family
    }
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    # Filter the relevant columns
    df = df[
        [
            "training_args.learning_rate",
            "training_args.beta",
            "data_regime",
            "num_ref",
            f"online_eval/eval_split/{REPORTING}/avg_reward_mean",
            f"online_eval/eval_split/{REPORTING}/avg_reward_std",
        ]
    ]

    df = df[
        df["num_ref"].isin(
            [
                1,
                # 2,
                3,
                # 6,
                10,
                # 18,
                25,
                # 36,
                50,
            ]
        )
    ]

    for i, regime in enumerate(["offpolicy2random", "offline"]):
        regime_df = df[df["data_regime"] == regime]

        for beta_idx, beta in enumerate(
            np.sort(regime_df["training_args.beta"].unique())
        ):
            regime_beta_df = regime_df[regime_df["training_args.beta"] == beta]
            regime_beta_df = regime_beta_df.sort_values(
                by="num_ref",
                ascending=False,
            )

            color = colors_dicr[regime][beta]

            if regime == "offline":
                extra_space_in_legend = ""
            else:
                extra_space_in_legend = ""

            # Plot the results
            regime_label = "off-policy" if "offpolicy" in regime else regime
            ax[beta_idx].errorbar(
                regime_beta_df["num_ref"],
                regime_beta_df[f"online_eval/eval_split/{REPORTING}/avg_reward_mean"],
                yerr=regime_beta_df[
                    f"online_eval/eval_split/{REPORTING}/avg_reward_std"
                ],
                fmt="o-",
                label=rf"{regime_label}, {extra_space_in_legend}$\beta$={beta}",
                color=color,
                lw=3.5,  # thicker main line
                elinewidth=1.2,  # thinner error bars
                capsize=3,  # optional: small caps on the error bars
                capthick=1.2,  # ...and make them as thin as elinewidth
            )
    # format the plots
    for ax_i in range(ax.shape[0]):
        ax[ax_i].set_xlabel("Number of references")
        ax[ax_i].set_ylabel("Reward")
        ax[ax_i].set_xscale("log")
        ax[ax_i].legend(loc="lower right")

        # ── 1. dashed, light-gray grid ─────────────────────────────────────────────
        ax[ax_i].grid(
            which="both",
            linestyle="--",
            linewidth=0.6,
            color="lightgray",
            alpha=0.8,
        )
    # --------------------------------------------------------------------------
    plt.tight_layout()
    out_path = Path(__file__).parent / f"scaling"
    out_path.mkdir(exist_ok=True)
    plt.savefig(
        out_path / f"num-ref-reward-scaling-{lr_name}.png",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    parsed_results_file = "neurips-2025-chat-scaling-20250718-202554.csv"
    out_path_csv = Path("outputs/shared/parsed-results") / f"{parsed_results_file}"
    print(f"Loading existing DataFrame from: {out_path_csv.resolve()}")
    df = pd.read_csv(out_path_csv)

    model = "llama"
    dataset = "magpieair"
    df = df[df["wandb.run_name"].str.match(f"{model}-.*-{dataset}-.*")]

    # Best checkpoint per run.
    best_indices = df.groupby("wandb.run_name")[
        f"online_eval/eval_split/{SELECTION}/avg_reward_mean"
    ].idxmax()
    df = df.loc[best_indices]

    # Extract parameters from the run name
    df["data_regime"] = df["wandb.run_name"].str.extract(
        r"-(offline|offpolicy2best|offpolicy2random)-"
    )
    df["num_ref"] = df["wandb.run_name"].str.extract(r"-numref(\d+)-").astype(int)

    # Main paper figure.
    df_main = df[
        (
            (df["training_args.learning_rate"] == 1e-7)
            & (df["training_args.beta"] == 0.1)
            | (df["training_args.learning_rate"] == 3e-7)
            & (df["training_args.beta"] == 0.01)
            | (df["training_args.learning_rate"] == 1e-6)
            & (df["training_args.beta"] == 0.003)
        )
    ]

    # Appendix figures.
    ## Best lr
    best_lrs = df.groupby(["data_regime", "training_args.beta", "num_ref"])[
        f"online_eval/eval_split/{SELECTION}/avg_reward_mean"
    ].idxmax()
    df_best = df.loc[best_lrs]

    ## lrs
    df_1e_6 = df[df["training_args.learning_rate"] == 1e-6]
    df_3e_7 = df[df["training_args.learning_rate"] == 3e-7]
    df_1e_7 = df[df["training_args.learning_rate"] == 1e-7]

    dfs = [df_main, df_best, df_1e_7, df_3e_7, df_1e_6]
    lr_names = ["main", "best", "1e-7", "3e-7", "1e-6"]
    for df, lr_name in zip(dfs, lr_names):
        plot_scaling(df, lr_name)
