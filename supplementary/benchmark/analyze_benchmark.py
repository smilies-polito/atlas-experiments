import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ["wall_time_s", "cpu_time_s", "peak_delta_gb"]
N_BOOT = 2000
SEED = 0


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def summarise(df):
    g = df.groupby(["function", "n_cells"])
    out = g[METRICS].agg(["median", q1, q3])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    for m in METRICS:
        out[f"{m}_cv"] = g[m].std() / g[m].mean()
    out["cpu_per_wall"] = g["cpu_time_s"].median() / g["wall_time_s"].median()
    out["n_runs"] = g.size()
    return out.reset_index()


def loglog_slope(df, metric, rng):
    """Slope of log(metric) ~ log(n_cells), bootstrap resampling runs within each size."""
    d = df[df[metric] > 0]
    if d["n_cells"].nunique() < 3:
        return np.nan, np.nan, np.nan
    x, y = np.log(d["n_cells"].to_numpy()), np.log(d[metric].to_numpy())
    slope = np.polyfit(x, y, 1)[0]
    groups = [np.flatnonzero(d["n_cells"].to_numpy() == n) for n in np.unique(d["n_cells"])]
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, lo, hi


def scaling(df, label):
    rng = np.random.default_rng(SEED)
    rows = []
    for fn, d in df.groupby("function"):
        for m in METRICS:
            s, lo, hi = loglog_slope(d, m, rng)
            rows.append({"function": fn, "metric": m, "subset": label,
                         "slope": s, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df["status"] == "ok"].copy()
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.dirname(os.path.abspath(path))

    summary = summarise(df)
    summary.to_csv(os.path.join(outdir, f"summary_{name}.tsv"), sep="\t", index=False)

    df["wall_rel_dev"] = df["wall_time_s"] / df.groupby(["function", "n_cells"])["wall_time_s"].transform("median") - 1
    busy = df["loadavg_1m_at_start"] > df["cpus_available"]
    print(f"runs started with load > allotted cores: {busy.sum()}/{len(df)}")
    print("\nSpearman correlation, load at start vs relative wall-time deviation:")
    for fn, d in df.groupby("function"):
        if d["wall_time_s"].median() < 0.5:   
            continue
        rho, p = spearmanr(d["loadavg_1m_at_start"], d["wall_rel_dev"])
        print(f"  {fn:32s} rho={rho:+.2f}  p={p:.3f}  n={len(d)}")

    quiet_threshold = df["loadavg_1m_at_start"].median()
    quiet = df[df["loadavg_1m_at_start"] <= quiet_threshold]
    scal = pd.concat([scaling(df, "all_runs"),
                      scaling(quiet, f"load<={quiet_threshold:.1f}")], ignore_index=True)
    scal.to_csv(os.path.join(outdir, f"scaling_{name}.tsv"), sep="\t", index=False)

    print("\nLog-log scaling exponents (1 = linear, 2 = quadratic), 95% bootstrap CI:")
    piv = scal.assign(txt=scal.apply(lambda r: f"{r.slope:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1))
    print(piv.pivot_table(index=["function", "metric"], columns="subset", values="txt", aggfunc="first").to_string())

    print("\nMedian CPU time / wall time (parallel efficiency is this divided by n_jobs):")
    print(summary.pivot(index="n_cells", columns="function", values="cpu_per_wall").round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
