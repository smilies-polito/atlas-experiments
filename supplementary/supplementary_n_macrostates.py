import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = os.path.join(os.getcwd(), "output", "mouse_hair", "results.tsv")
OUTDIR = os.path.join(os.getcwd(), "figures")
os.makedirs(OUTDIR, exist_ok=True)

NCOLS = 3

METRIC_DICT = {"spearman_stat_pseudotime": (-1,1),
                "kendall_stat_pseudotime": (-1,1),
                "temporal_state_score": (0,1),
                "tts": (0,1),
                "ttp": (0,1),
                "tsr": (0,1),
                "ttc": (0,1),
                "fate_index_stat": (-1,1),
                "terminal_enrichment": (-1,1),
                "terminal_silhouette_soft": (-1,1),
                "terminal_silhouette_pse": (-1, 1),
                "spearman_stat_SHE": (-1,1),
                "spearman_stat_KLD": (-1,1),
                "pearson_stat_SHE": (-1,1),
                "pearson_stat_KLD": (-1,1),
                "totipotent": (0, np.sqrt(np.log(2))),
                "multipotent": (0, np.sqrt(np.log(2))),
                "committed": (0, np.sqrt(np.log(2))),
                }

# nomi delle colonne del tsv -> nomi in METRIC_DICT
METRIC_ALIASES = {
    "pearson_stat_KL": "pearson_stat_KLD",
    "spearman_stat_KL": "spearman_stat_KLD",
    "silhouette_soft": "terminal_silhouette_soft",
    "silhouette_hard": "terminal_silhouette_pse",
}

def parse_code(code):
    if code.startswith("rna_cellrank"):
        method = "RNA-CellRank"
        n_macrostates = code.split(".")[0].split("_")[-1]
    elif code.startswith("cellrank"):
        method = "CellRank"
        n_macrostates = code.split(".")[0].split("_")[-1]
    elif code.startswith("rna_palantir"):
        method = "RNA-Palantir"
        n_macrostates = None
    elif code.startswith("palantir"):
        method = "Palantir"
        n_macrostates = None
    else:
        method = "Unknown"
        n_macrostates = None
    return method, n_macrostates

def add_ci_pval(ax, data, metric, line):
    # CI come banda colorata, se nel tsv ci sono le colonne ciL / ciH
    ciL, ciH = metric.replace("_stat", "_ciL"), metric.replace("_stat", "_ciH")
    if ciL != metric and ciL in data.columns and ciH in data.columns:
        ax.fill_between(
            data["n_macrostates"],
            data[ciL],
            data[ciH],
            color=line.get_color(),
            alpha=0.2,
        )
    # marker vuoto (bianco) sui punti NON significativi (p-value >= 0.05)
    pval = metric.replace("_stat", "_pval")
    if pval != metric and pval in data.columns:
        ns = data[data[pval] >= 0.05]
        ax.scatter(
            ns["n_macrostates"],
            ns[metric],
            facecolor="white",
            edgecolor=line.get_color(),
            linewidth=1.5,
            zorder=3,
        )

if __name__=="__main__":

    df = pd.read_csv(INPUT, sep="\t")
    df[["method", "n_macrostates"]] = df["code"].apply(
            lambda x: pd.Series(parse_code(x))
    )
    df["n_macrostates"] = pd.to_numeric(df["n_macrostates"])

    metrics = {
        "pearson_stat_KL": "Pearson correlation — KL",
        "spearman_stat_KL": "Spearman correlation — KL",
        "pearson_stat_SHE": "Pearson correlation — SHE",
        "spearman_stat_SHE": "Spearman correlation — SHE",
        "fate_index_stat": "Fate index",
        "pseudotime_enrichment": "Pseudotime enrichment",
        "silhouette_soft": "Silhouette — soft",
        "silhouette_hard": "Silhouette — hard",
    }

    ncols = NCOLS
    nrows = math.ceil(len(metrics) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        constrained_layout=True,
    )

    axes = axes.flatten()
    for ax, (metric, ylabel) in zip(axes, metrics.items()):
        cr = df[
            (df["method"] == "CellRank")
            & df["n_macrostates"].notna()
        ].sort_values("n_macrostates")

        line, = ax.plot(
            cr["n_macrostates"],
            cr[metric],
            marker="o",
            linewidth=2,
            label="ATLAS CellRank-strategy",
        )
        add_ci_pval(ax, cr, metric, line)

        rna_cr = df[
            (df["method"] == "RNA-CellRank")
            & df["n_macrostates"].notna()
        ].sort_values("n_macrostates")

        line, = ax.plot(
            rna_cr["n_macrostates"],
            rna_cr[metric],
            marker="o",
            linewidth=2,
            label="CellRank",
        )
        add_ci_pval(ax, rna_cr, metric, line)

        ylim = METRIC_DICT.get(metric, METRIC_DICT.get(METRIC_ALIASES.get(metric)))
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(ylabel)
        ax.set_xlabel("# macrostates")
        ax.set_xticks(range(4, 11))
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.remove()

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.savefig(
        os.path.join(OUTDIR, "metrics_vs_macrostates.png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(OUTDIR, "metrics_vs_macrostates.pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)
