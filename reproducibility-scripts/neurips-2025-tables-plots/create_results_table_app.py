import math
import re
from pathlib import Path

import pandas as pd  # or from tqdm import tqdm

SELECTION = "selection"
REPORTING = "reporting"

_LOSSES_ORDER = ["dpo", "simpo", "rebel", "qrpo"]
_LOSSES_NAME_MAP = {
    "dpo": "DPO",
    "simpo": "SimPO",
    "rebel": "REBEL",
    "qrpo": "QRPO",
}

_CHAT_COL_FORMAT = {
    # core metrics
    f"online_eval/eval_split/{SELECTION}/avg_reward_mean": ("{:.4f}", "{:.4f}"),
    f"online_eval/eval_split/{REPORTING}/avg_reward_mean": ("{:.4f}", "{:.4f}"),
    f"online_eval/eval_split/{SELECTION}/lc_lcmodel_avg_reward_mean": (
        "{:.4f}",
        "{:.4f}",
    ),
    f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_mean": (
        "{:.4f}",
        "{:.4f}",
    ),
    # config
    "training_args.beta": ("{:0.0e}", None),
    "training_args.learning_rate": ("{:0.0e}", None),
    "checkpoint_num": ("{:.0f}", None),
}

_CODE_COL_FORMAT = {
    # core metrics
    f"online_eval/eval_split/{SELECTION}/avg_reward_mean": ("{:.3f}", "{:.3f}"),
    f"online_eval/eval_split/{REPORTING}/avg_reward_mean": ("{:.3f}", "{:.3f}"),
    f"online_eval/eval_split/{SELECTION}/lc_lcmodel_avg_reward_mean": (
        "{:.3f}",
        "{:.3f}",
    ),
    f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_mean": (
        "{:.3f}",
        "{:.3f}",
    ),
    # config
    "training_args.beta": ("{:0.0e}", None),
    "training_args.learning_rate": ("{:0.0e}", None),
    "checkpoint_num": ("{:.0f}", None),
}

_MODEL_LABEL = {
    "llama": r"Llama 8B Tülu 3 SFT",
    "mistral": r"Mistral 7B Instruct v0.2",
}

_REGIME_LABEL = {
    "offline": "offline",
    "offpolicy2best": "offpolicy--best",
    "offpolicy2random": "offpolicy--random",
    "offpolicy10random": "offpolicy--random",
}

_SET_ORDER = [
    ("nosft", "offline"),
    ("nosft", "offpolicy2best"),
    ("nosft", "offpolicy2random"),
    ("sft", "offline"),
    ("sft", "offpolicy2best"),
    ("sft", "offpolicy2random"),
    ("sft", "offpolicy10random"),
    ("nosft", "offpolicy10random"),
]

_PLACEHOLDER = r"\textemdash{}"


def _format(mean, std, *, colname, code_or_chat, underline=False):
    """
    Formats a single numeric cell.

    highlight=True  ⇒  mean part is bold
    underline=True  ⇒  mean part is underlined
    Both can be true simultaneously (underline wraps the bold).
    """

    if pd.isna(mean):
        return _PLACEHOLDER

    if code_or_chat == "chat":
        _COL_FORMAT = _CHAT_COL_FORMAT
    elif code_or_chat == "code":
        _COL_FORMAT = _CODE_COL_FORMAT
    else:
        raise ValueError(f"Unknown code_or_chat value: {code_or_chat}")

    fmt_mean, fmt_err = _COL_FORMAT.get(colname)

    # ---------- pretty-printing for β --------------------------
    if colname == "training_args.beta" and 1e-3 <= abs(mean) < 1e3:
        txt = f"{mean:g}"
        if underline:
            txt = rf"\underline{{{txt}}}"
        return txt

    # ---------- branch: no ±error shown --------------------------------
    if fmt_err is None:
        txt = fmt_mean.format(mean)
        # clean up scientific notation (e-04 → e-4)
        txt = re.sub(
            r"e([+-])?0*(\d+)",
            lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2),
            txt,
        )
        if underline:
            txt = rf"\underline{{{txt}}}"
        return txt

    # ---------- branch: value ± error ----------------------------------
    mean_txt = fmt_mean.format(mean)
    err_txt = fmt_err.format(std) if not pd.isna(std) else _PLACEHOLDER

    if underline:
        mean_txt = rf"\underline{{{mean_txt}}}"

    return rf"\thinValue{{{mean_txt}}}{{{err_txt}}}"


# --------------------------------------------------------------------------
# main table builder -------------------------------------------------------
# --------------------------------------------------------------------------


def generate_latex_table(
    df: pd.DataFrame,
    model: str,
    dataset: str,
    chat_or_code: str,
):
    # ------------------------------------------------------------------ #
    # 1. metric list (internal IDs)                                      #
    # ------------------------------------------------------------------ #
    if chat_or_code == "chat":
        metrics = ["Reward", "LC-Reward", "$\\beta$", "LR", "Step"]
    elif chat_or_code == "code":
        metrics = ["Reward", "$\\beta$", "LR", "Step"]

    # display names for the header
    METRIC_LABEL = {
        "Reward": r"Reward (valid/test)",
        "LC-Reward": r"LC-Reward (valid/test)",
    }

    n_metrics = len(metrics)

    # convenient column aliases
    SEL_REWARD = f"online_eval/eval_split/{SELECTION}/avg_reward_mean"
    SEL_REWARD_S = f"online_eval/eval_split/{SELECTION}/avg_reward_std"
    REP_REWARD = f"online_eval/eval_split/{REPORTING}/avg_reward_mean"
    REP_REWARD_S = f"online_eval/eval_split/{REPORTING}/avg_reward_std"

    SEL_LCREWARD = f"online_eval/eval_split/{SELECTION}/lc_lcmodel_avg_reward_mean"
    SEL_LCREWARD_S = f"online_eval/eval_split/{SELECTION}/lc_lcmodel_avg_reward_std"
    REP_LCREWARD = f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_mean"
    REP_LCREWARD_S = f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_std"

    # Best over HP for each loss and distribution regime
    displayed = {}
    for loss in _LOSSES_ORDER:
        subdf = df[df["training_args.loss_type"].str.lower() == loss]
        for sft_flag, reg in _SET_ORDER:
            rows = subdf[
                (subdf["sftnosft"] == sft_flag) & (subdf["data_regime"] == reg)
            ]
            row = rows.loc[rows[SEL_REWARD].idxmax()] if not rows.empty else None
            displayed[(loss, sft_flag, reg)] = row

    # ------------------------------------------------------------------ #
    # 3. LaTeX prologue                                                  #
    # ------------------------------------------------------------------ #
    colspec = "ll" + "c" * n_metrics
    model_lbl = _MODEL_LABEL.get(model.lower(), model.capitalize())
    ds_lbl = {
        "leetcode": "LeetCodeDataset",
        "magpieair": "Magpie-Air",
        "ultrafeedback": "UltraFeedback",
    }.get(dataset, dataset)

    lines = []
    add = lines.append
    add(r"\begin{table}[!htb]")
    add(r"\centering")
    add(f"\\caption{{{model_lbl}, {ds_lbl} dataset.}}")
    add(f"\\label{{tab:{model}-{dataset}}}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{" + colspec + r"}")
    add(r"\toprule")

    header_cells = [r"\textbf{Method}", r"Setting"]
    for m in metrics:
        label = METRIC_LABEL.get(m, m)  # rename two main columns
        header_cells.append(label)
    add(" & ".join(header_cells) + r" \\")

    # ------------------------------------------------------------------ #
    # 4. render method blocks                                            #
    # ------------------------------------------------------------------ #
    best_models = []
    shortlist_models = []
    shortlist_x2_models = []
    for loss in _LOSSES_ORDER:
        add(r"\midrule")
        vis_rows = [
            displayed[(loss, s, r)]
            for s, r in _SET_ORDER
            if displayed[(loss, s, r)] is not None
        ]
        if len(vis_rows) == 0:
            continue
        mdf = pd.DataFrame(vis_rows)
        best_sel_reward_id = mdf[SEL_REWARD].idxmax(skipna=True)
        best_sel_reward = mdf.loc[best_sel_reward_id][SEL_REWARD]
        best_sel_reward_s = mdf.loc[best_sel_reward_id][SEL_REWARD_S]

        def is_shortlist(best_sel_reward, best_sel_reward_s, s_mean, s_std, k):
            return pd.notna(s_mean) and (best_sel_reward - s_mean) <= k * math.sqrt(
                s_std**2 + best_sel_reward_s**2
            )

        def is_reward_shortlist(best_sel_reward, best_sel_reward_s, s_mean, s_std):
            return is_shortlist(best_sel_reward, best_sel_reward_s, s_mean, s_std, k=2)

        def is_lc_shortlist(best_sel_reward, best_sel_reward_s, s_mean, s_std):
            return is_shortlist(
                best_sel_reward, best_sel_reward_s, s_mean, s_std, k=0.8
            )

        # Shortlist based on reward and find best LC-Reward to narrow the shortlist.
        best_sel_lc = -1
        best_self_lc_s = None
        for sft_flag, reg in _SET_ORDER:
            row = displayed[(loss, sft_flag, reg)]
            if row is None:
                continue
            if is_reward_shortlist(
                best_sel_reward,
                best_sel_reward_s,
                row[SEL_REWARD],
                row.get(SEL_REWARD_S),
            ):
                shortlist_models.append(row)
                if row[SEL_LCREWARD] >= best_sel_lc:
                    best_sel_lc = row[SEL_LCREWARD]
                    best_self_lc_s = row.get(SEL_LCREWARD_S)

        # narrow the shortlist based on LC.
        best_reward_shortlist_x2 = -1
        for sft_flag, reg in _SET_ORDER:
            row = displayed[(loss, sft_flag, reg)]
            if row is None:
                continue
            if is_reward_shortlist(
                best_sel_reward,
                best_sel_reward_s,
                row[SEL_REWARD],
                row.get(SEL_REWARD_S),
            ) and is_lc_shortlist(
                best_sel_lc, best_self_lc_s, row[SEL_LCREWARD], row.get(SEL_LCREWARD_S)
            ):
                shortlist_x2_models.append(row)
                if row[SEL_REWARD] >= best_reward_shortlist_x2:
                    best_reward_shortlist_x2 = row[SEL_REWARD]

        block_rows, first = (
            len(
                [
                    1
                    for sft_flag, reg in _SET_ORDER
                    if displayed[(loss, sft_flag, reg)] is not None
                ]
            ),
            True,
        )
        for sft_flag, reg in _SET_ORDER:
            row = displayed[(loss, sft_flag, reg)]
            if row is None:
                continue

            s_mean, r_mean = row[SEL_REWARD], row[REP_REWARD]
            s_std, r_std = row.get(SEL_REWARD_S), row.get(REP_REWARD_S)
            ul_sel_reward = is_reward_shortlist(
                best_sel_reward, best_sel_reward_s, s_mean, s_std
            )

            s_lc_mean, r_lc_mean = row[SEL_LCREWARD], row[REP_LCREWARD]
            s_lc_std, r_lc_std = row.get(SEL_LCREWARD_S), row.get(REP_LCREWARD_S)
            ul_sel_lc = ul_sel_reward and is_lc_shortlist(
                best_sel_lc, best_self_lc_s, s_lc_mean, s_lc_std
            )

            vals = []
            for m in metrics:
                if m == "Reward":
                    s_txt = _format(
                        s_mean,
                        s_std,
                        colname=SEL_REWARD,
                        code_or_chat=chat_or_code,
                        underline=ul_sel_reward,
                    )
                    r_txt = _format(
                        r_mean,
                        r_std,
                        colname=REP_REWARD,
                        code_or_chat=chat_or_code,
                        underline=False,
                    )
                    vals.append(f"{s_txt} / {r_txt}")

                elif m == "LC-Reward":
                    s_txt = _format(
                        s_lc_mean,
                        s_lc_std,
                        colname=SEL_LCREWARD,
                        code_or_chat=chat_or_code,
                        underline=ul_sel_lc,
                    )
                    r_txt = _format(
                        r_lc_mean,
                        r_lc_std,
                        colname=REP_LCREWARD,
                        code_or_chat=chat_or_code,
                        underline=False,
                    )
                    vals.append(f"{s_txt} / {r_txt}")

                elif m == "$\\beta$":
                    vals.append(
                        _format(
                            row["training_args.beta"],
                            None,
                            colname="training_args.beta",
                            code_or_chat=chat_or_code,
                            underline=False,
                        )
                    )
                elif m == "LR":
                    vals.append(
                        _format(
                            row["training_args.learning_rate"],
                            None,
                            colname="training_args.learning_rate",
                            code_or_chat=chat_or_code,
                            underline=False,
                        )
                    )
                elif m == "Step":
                    vals.append(
                        _format(
                            row["checkpoint_num"],
                            None,
                            colname="checkpoint_num",
                            code_or_chat=chat_or_code,
                            underline=False,
                        )
                    )

            label = _REGIME_LABEL[reg]
            if sft_flag == "sft":
                label = r"SFT $\rightarrow$ " + label
            if (
                ul_sel_reward
                and ul_sel_lc
                and best_reward_shortlist_x2 <= row[SEL_REWARD]
            ):
                label += "*"
                best_models.append(row)

            row_cells = []
            if first:
                row_cells.append(
                    rf"\multirow{{{block_rows}}}{{*}}{{\textbf{{{_LOSSES_NAME_MAP[loss]}}}}}"
                )
                first = False
            else:
                row_cells.append("")
            row_cells.append(label)
            row_cells.extend(vals)
            add(" & ".join(row_cells) + r" \\")

    add(r"\bottomrule")
    add(r"\end{tabular}}")
    add(r"\end{table}")

    return (
        "\n".join(lines),
        pd.DataFrame(best_models, columns=df.columns),
        pd.DataFrame(shortlist_models, columns=df.columns),
        pd.DataFrame(shortlist_x2_models, columns=df.columns),
    )


def print_tables_for_all_models(
    all_df: pd.DataFrame, datasets, models, chat_or_code="chat"
):
    if chat_or_code == "chat":
        filter_length = True
        add_alpaca = True
    elif chat_or_code == "code":
        filter_length = False
        add_alpaca = False
    else:
        raise ValueError(f"Unknown chat_or_code value: {chat_or_code}")
    for dataset in datasets:
        for model in models:
            mask = all_df["wandb.run_name"].str.match(f"{model}-.*-{dataset}-.*")
            filtered = all_df.loc[mask].copy()

            # convenience columns
            filtered["sftnosft"] = filtered["wandb.run_name"].str.extract(
                r"-(sft|nosft)-"
            )
            filtered["data_regime"] = filtered["wandb.run_name"].str.extract(
                r"-(offline|offpolicy2best|offpolicy2random|offpolicy10random)-"
            )

            # remove degenerate models.
            if filter_length:
                keep = (
                    filtered[
                        f"online_eval/eval_split/{SELECTION}/avg_reward_tokens_len_mean"
                    ]
                    <= 1000
                )
                keep = keep & (
                    filtered[
                        f"online_eval/eval_split/{REPORTING}/avg_reward_tokens_len_mean"
                    ]
                    <= 1000
                )
                filtered = filtered.loc[keep]

            # Uncomment for Base and Base+SFT models.
            # mask = filtered["checkpoint_num"] == 0
            # mask &= filtered["sftnosft"] == "sft"
            # filtered = filtered[mask]

            (
                latex_code,
                best_models,
                shortlist_models,
                shortlist_x2_models,
            ) = generate_latex_table(filtered, model, dataset, chat_or_code)

            print("\n" + "% " + "=" * 50)
            print(f"% {model.upper()} – {dataset}")
            print("% " + "=" * 50 + "\n")
            print(latex_code)
            print()

            keys = [
                "wandb.run_name",
                "training_args.loss_type",
                "training_args.beta",
                "training_args.learning_rate",
                "checkpoint_num",
                f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_mean",
                f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_std",
                f"online_eval/eval_split/{REPORTING}/avg_reward_mean",
                f"online_eval/eval_split/{REPORTING}/avg_reward_std",
                f"online_eval/eval_split/{REPORTING}/avg_reward_tokens_len_mean",
                f"online_eval/eval_split/{REPORTING}/avg_reward_tokens_len_std",
                f"online_eval/eval_split/{SELECTION}/lc_lcmodel_avg_reward_mean",
                f"online_eval/eval_split/{SELECTION}/lc_lcmodel_avg_reward_std",
                f"online_eval/eval_split/{SELECTION}/avg_reward_mean",
                f"online_eval/eval_split/{SELECTION}/avg_reward_std",
                f"online_eval/eval_split/{SELECTION}/avg_reward_tokens_len_mean",
                f"online_eval/eval_split/{SELECTION}/avg_reward_tokens_len_std",
            ]

            if add_alpaca:
                keys += [
                    "alpaca/length_controlled_winrate",
                    "alpaca/lc_standard_error",
                    "alpaca/win_rate",
                    "alpaca/standard_error",
                    "alpaca/avg_length",
                    "online_eval/alpaca_eval/reporting/avg_reward_mean",
                    "online_eval/alpaca_eval/reporting/avg_reward_tokens_len_mean",
                    "online_eval/alpaca_eval/reporting/avg_reward_tokens_len_std",
                ]

            best_models = best_models[keys]

            best_models[
                f"online_eval/eval_split/{REPORTING}/avg_reward_mean"
            ] = best_models[
                f"online_eval/eval_split/{REPORTING}/avg_reward_mean"
            ].apply(
                lambda x: f"{x:.4f}"
            )
            best_models[
                f"online_eval/eval_split/{REPORTING}/avg_reward_std"
            ] = best_models[f"online_eval/eval_split/{REPORTING}/avg_reward_std"].apply(
                lambda x: f"{x:.4f}"
            )
            best_models[
                f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_mean"
            ] = best_models[
                f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_mean"
            ].apply(
                lambda x: f"{x:.4f}"
            )
            best_models[
                f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_std"
            ] = best_models[
                f"online_eval/eval_split/{REPORTING}/lc_lcmodel_avg_reward_std"
            ].apply(
                lambda x: f"{x:.4f}"
            )
            if add_alpaca:
                best_models["alpaca/length_controlled_winrate"] = best_models[
                    "alpaca/length_controlled_winrate"
                ].apply(lambda x: f"{x:.1f}")
                best_models["alpaca/lc_standard_error"] = best_models[
                    "alpaca/lc_standard_error"
                ].apply(lambda x: f"{x:.1f}")
                best_models["alpaca/win_rate"] = best_models["alpaca/win_rate"].apply(
                    lambda x: f"{x:.1f}"
                )
                best_models["alpaca/standard_error"] = best_models[
                    "alpaca/standard_error"
                ].apply(lambda x: f"{x:.1f}")

            shortlist_models = shortlist_models[keys]
            shortlist_x2_models = shortlist_x2_models[keys]

            print("\nBest models:")
            print(best_models.iloc[:100].to_string())
            print("\nShortlist models:")
            print(shortlist_models.iloc[:100].to_string())
            print("\nShortlist x2 models:")
            print(shortlist_x2_models.iloc[:100].to_string())


if __name__ == "__main__":
    parsed_results_file = "neurips-2025-chat-baseline-20250718-202520.csv"
    out_path_csv = Path("outputs/shared/parsed-results") / f"{parsed_results_file}"
    df = pd.read_csv(out_path_csv)
    print_tables_for_all_models(
        df, ["magpieair", "ultrafeedback"], ["llama", "mistral"], chat_or_code="chat"
    )

    parsed_results_file = "neurips-2025-code-baseline-20250718-202559.csv"
    out_path_csv = Path("outputs/shared/parsed-results") / f"{parsed_results_file}"
    df = pd.read_csv(out_path_csv)
    print_tables_for_all_models(df, ["leetcode"], ["llama"], chat_or_code="code")
