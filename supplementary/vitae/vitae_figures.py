import os
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WD = os.getcwd()
DATA_DIR = os.path.join(WD, "output", "embryonic_mouse_brain", "vitae_input")
FIG_DIR = os.path.join(WD, "figures")

# colori fissi: la modalità ha lo stesso colore in tutte le figure
MOD_COLORS = {"RNA": "#2a78d6", "ACT": "#eb6834"}
CMAP_SEQ = "viridis"
NA_COLOR = "#d9d9d9"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
    "xtick.color": "#555555", "ytick.color": "#555555",
})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _scatter_continuous(ax, xy, v, title, cmap=CMAP_SEQ, vmin=None, vmax=None, label=None):
    nan = np.isnan(v)
    ax.scatter(xy[nan, 0], xy[nan, 1], s=2, c=NA_COLOR, linewidths=0, rasterized=True)
    o = np.argsort(v[~nan])
    s = ax.scatter(xy[~nan][o, 0], xy[~nan][o, 1], s=2, c=v[~nan][o], cmap=cmap,
                   vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9, loc="left")
    _clean(ax)
    cb = plt.colorbar(s, ax=ax, shrink=0.7, pad=0.01)
    if label:
        cb.set_label(label, fontsize=8)
    return s


def _vertex_order(adata):
    vs = list(adata.uns["vitae_vertices"])
    med = (adata.obs.groupby(adata.obs["vitae_new_clustering"].astype(str))["pseudotime"]
           .median().reindex(vs))
    return list(med.sort_values(na_position="last").index)


def _vertex_xy(adata):
    xy = adata.obsm["X_vitae_umap"]
    lab = adata.obs["vitae_new_clustering"].astype(str).values
    return {v: xy[lab == v].mean(0) for v in adata.uns["vitae_vertices"] if (lab == v).any()}


def _draw_graph(ax, adata, directed: bool):
    key = "vitae_directed_edges" if directed else "vitae_backbone_edges"
    edges = pd.DataFrame(adata.uns[key])
    pos = _vertex_xy(adata)
    if len(edges) == 0:
        return
    w = edges["score"].astype(float).values
    w = 1 + 3 * (w - w.min()) / (np.ptp(w) + 1e-12)
    seen = set()
    for (a, b), lw in zip(edges[["from", "to"]].values, w):
        if a not in pos or b not in pos or (not directed and (b, a) in seen):
            continue
        seen.add((a, b))
        if directed:
            ax.annotate("", xy=pos[b], xytext=pos[a], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", lw=lw, color="black",
                                        shrinkA=6, shrinkB=6, mutation_scale=10))
        else:
            ax.plot(*np.c_[pos[a], pos[b]], color="black", lw=lw, zorder=4,
                    solid_capstyle="round")
    for v, p in pos.items():
        ax.scatter(*p, s=110, c="white", edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(v, p, xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=7, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def fig_latent(adata, tag, cluster_key):
    adata.uns["condition_colors"] = [MOD_COLORS[c] for c in adata.obs["condition"].cat.categories]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, key, title in [(axes[0, 0], "condition", "Modalità (RNA vs ACT)"),
                           (axes[0, 1], cluster_key, "Tipo cellulare")]:
        sc.pl.embedding(adata, basis="vitae_umap", color=key, ax=ax, show=False,
                        frameon=False, title=title, size=6, legend_fontsize=7)
    xy = adata.obsm["X_vitae_umap"]
    _scatter_continuous(axes[1, 0], xy, adata.obs["pseudotime"].values.astype(float),
                        "Pseudotime")
    _scatter_continuous(axes[1, 1], xy, adata.obs["projection_uncertainty"].values.astype(float),
                    "Uncertainty about projection upon the graph", cmap="Reds")
    fig.suptitle("VITAE – integrated latent space (UMAP)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    _save(fig, f"{tag}_fig1_latent")


def fig_trajectory(adata, tag, cluster_key):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.embedding(adata, basis="vitae_umap", color=cluster_key, ax=axes[0], show=False,
                    frameon=False, size=6, legend_fontsize=7, alpha=0.5,
                    title="Backbone (width = edge score)")
    _draw_graph(axes[0], adata, directed=False)
    _scatter_continuous(axes[1], adata.obsm["X_vitae_umap"],
                        adata.obs["pseudotime"].values.astype(float),
                        "Root oriented trajectory")
    _draw_graph(axes[1], adata, directed=True)
    fig.tight_layout()
    _save(fig, f"{tag}_fig2_trajectory")


def _pcx_frame(adata):
    return pd.DataFrame(np.asarray(adata.obsm["vitae_pc_x"]), index=adata.obs_names,
                        columns=list(adata.uns["vitae_states"]))


def _ordered_states(adata, P, min_mass=0.01):
    order = _vertex_order(adata)
    rank = {v: i for i, v in enumerate(order)}
    keep = P.columns[P.mean(0) >= min_mass / 10]
    keep = [s for s in keep if (P[s] >= min_mass).any()]

    def key(s):
        if " — " in s:
            a, b = s.split(" — ")
            return (1, min(rank[a], rank[b]), max(rank[a], rank[b]))
        return (0, rank[s], 0)
    return sorted(keep, key=key)


def fig_likelihood_umap(adata, tag):
    P = _pcx_frame(adata).values
    vertex_cols = [i for i, s in enumerate(adata.uns["vitae_states"]) if " — " not in s]
    conf = P.max(1)
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1) / np.log(P.shape[1])
    on_edge = 1 - P[:, vertex_cols].sum(1)

    xy = adata.obsm["X_vitae_umap"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    _scatter_continuous(axes[0], xy, conf, "Posterior density  max_c p(c|Y)", vmin=0, vmax=1)
    _scatter_continuous(axes[1], xy, ent, "Normalised entropy fro p(c|Y)", cmap="magma_r",
                        vmin=0, vmax=max(0.05, np.nanpercentile(ent, 99)))
    _scatter_continuous(axes[2], xy, on_edge, "Probability of transitioning on an edge",
                        cmap="Blues", vmin=0, vmax=1)
    fig.suptitle("VITAE – likelihood (posterior likelihood)", x=0.02, y=1.03,
                 ha="left", fontsize=11)
    fig.tight_layout()
    _save(fig, f"{tag}_fig3_likelihood_umap")
    return pd.DataFrame({"pcx_max": conf, "pcx_entropy": ent, "pcx_on_edge": on_edge},
                        index=adata.obs_names)


def fig_likelihood_vertices(adata, tag, ncol=5):
    W = pd.DataFrame(np.asarray(adata.obsm["vitae_w_posterior"]), index=adata.obs_names,
                     columns=list(adata.uns["vitae_vertices"]))
    order = _vertex_order(adata)
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow), squeeze=False)
    xy = adata.obsm["X_vitae_umap"]
    for ax, v in zip(axes.flat, order):
        w = W[v].values
        o = np.argsort(w)
        s = ax.scatter(xy[o, 0], xy[o, 1], s=2, c=w[o], cmap=CMAP_SEQ, vmin=0, vmax=1,
                       linewidths=0, rasterized=True)
        ax.set_title(v, fontsize=9, loc="left")
        _clean(ax)
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.colorbar(s, ax=axes, shrink=0.5, label="weight on the vertex  E[w̃|Y]")
    fig.suptitle("VITAE – cell vertex position a posteriori (ordered accoridng pseudotime)",
                x=0.02, y=1.02, ha="left", fontsize=11)
    _save(fig, f"{tag}_fig4_likelihood_vertices")


def fig_likelihood_heatmap(adata, tag, cluster_key):
    P = _pcx_frame(adata)
    states = _ordered_states(adata, P)
    ct_order = (adata.obs.groupby(cluster_key, observed=True)["pseudotime"].median()
                .sort_values(na_position="last").index.tolist())
    mats = {}
    for mod in ["RNA", "ACT"]:
        m = adata.obs["condition"].values == mod
        mats[mod] = (P.loc[m, states].groupby(adata.obs.loc[m, cluster_key].values, observed=True)
                     .mean().reindex(ct_order))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.32 * len(states) * 2 + 3),
                                            0.35 * len(ct_order) + 2.2), sharey=True)
    for ax, mod in zip(axes, ["RNA", "ACT"]):
        im = ax.imshow(mats[mod].values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ct_order)))
        ax.set_yticklabels(ct_order, fontsize=8)
        n_vert = sum(" — " not in s for s in states)
        ax.axvline(n_vert - 0.5, color=MOD_COLORS[mod], lw=1.5)
        ax.set_title(f"Cellule {mod}", loc="left", fontsize=10, color=MOD_COLORS[mod])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="p(c | Y) media")
    fig.suptitle("Where VITAE locates every cell type: vertices (left from the line) "
                 "and edges (at the right)", x=0.02, y=1.02, ha="left", fontsize=11)
    _save(fig, f"{tag}_fig5_likelihood_heatmap")


def fig_likelihood_cells(adata, tag, cluster_key, n_max=3000, seed=0):
    P = _pcx_frame(adata)
    states = _ordered_states(adata, P)
    obs = adata.obs.copy()
    idx = np.arange(adata.n_obs)
    if len(idx) > n_max:
        idx = np.random.default_rng(seed).choice(idx, n_max, replace=False)
    sub = obs.iloc[idx].assign(_i=idx).sort_values("pseudotime", na_position="last")
    M = P.iloc[sub["_i"].values][states].values.T

    cts = sub[cluster_key].astype(str)
    ct_cats = list(adata.obs[cluster_key].astype("category").cat.categories)
    pal = adata.uns.get(f"{cluster_key}_colors",
                        plt.cm.tab20(np.linspace(0, 1, len(ct_cats))))
    ct_cmap = ListedColormap(pal[:len(ct_cats)])
    mod_cmap = ListedColormap([MOD_COLORS["RNA"], MOD_COLORS["ACT"]])

    fig = plt.figure(figsize=(11, 0.18 * len(states) + 2.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.5, 0.25, 0.25, 0.18 * len(states)],
                          width_ratios=[1, 0.02], hspace=0.08, wspace=0.02)
    a0 = fig.add_subplot(gs[0, 0])
    a0.plot(sub["pseudotime"].values, color="#333333", lw=1)
    a0.set_ylabel("pt", fontsize=7); a0.set_xticks([]); a0.set_xlim(0, len(sub) - 1)
    a1 = fig.add_subplot(gs[1, 0])
    a1.imshow([pd.Categorical(cts, categories=ct_cats).codes], aspect="auto", cmap=ct_cmap,
              vmin=0, vmax=len(ct_cats) - 1, interpolation="nearest")
    a1.set_yticks([0]); a1.set_yticklabels([cluster_key], fontsize=7); a1.set_xticks([])
    a2 = fig.add_subplot(gs[2, 0])
    a2.imshow([(sub["condition"].values == "ACT").astype(int)], aspect="auto", cmap=mod_cmap,
              vmin=0, vmax=1, interpolation="nearest")
    a2.set_yticks([0]); a2.set_yticklabels(["modality"], fontsize=7); a2.set_xticks([])
    a3 = fig.add_subplot(gs[3, 0])
    im = a3.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    a3.set_yticks(range(len(states))); a3.set_yticklabels(states, fontsize=6)
    a3.set_xlabel(f"cells ordered along pseudotime (n = {len(sub)})")
    a3.set_xticks([])
    for a in (a0, a1, a2, a3):
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, cax=fig.add_subplot(gs[3, 1]), label="p(c | Y)")
    h_mod = [plt.Line2D([], [], marker="s", ls="", color=MOD_COLORS[m], label=m) for m in MOD_COLORS]
    h_ct = [plt.Line2D([], [], marker="s", ls="", color=ct_cmap(i), label=c)
            for i, c in enumerate(ct_cats)]
    fig.legend(handles=h_mod, title="modality", loc="upper left", bbox_to_anchor=(0.92, 0.98),
               frameon=False, fontsize=7, title_fontsize=7)
    fig.legend(handles=h_ct, title=cluster_key, loc="upper left", bbox_to_anchor=(0.97, 0.98),
               frameon=False, fontsize=7, title_fontsize=7)
    fig.suptitle("VITAE – likelihood for every cell along pseudotime", x=0.02, y=0.99,
                 ha="left", fontsize=11)
    _save(fig, f"{tag}_fig6_likelihood_cells")


def fig_pseudotime_by_type(adata, tag, cluster_key):
    obs = adata.obs[[cluster_key, "condition", "pseudotime"]].dropna()
    obs[cluster_key] = obs[cluster_key].astype(str)
    order = (obs[obs["condition"] == "RNA"].groupby(cluster_key)["pseudotime"]
             .median().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(order) + 2), 3.8))
    width = 0.36
    for j, mod in enumerate(["RNA", "ACT"]):
        data = [obs.loc[(obs[cluster_key] == ct) & (obs["condition"] == mod), "pseudotime"].values
                for ct in order]
        pos = np.arange(len(order)) + (j - 0.5) * width
        keep = [k for k, d in enumerate(data) if len(d)]
        bp = ax.boxplot([data[k] for k in keep], positions=pos[keep], widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=MOD_COLORS[mod], edgecolor=MOD_COLORS[mod])
        for k in ("whiskers", "caps"):
            for line in bp[k]:
                line.set(color=MOD_COLORS[mod], lw=1)
        ax.plot([], [], color=MOD_COLORS[mod], lw=6, label=mod)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Pseudotime VITAE")
    ax.set_title("Pseudotime for every cell type (odered according to median RNA)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{tag}_fig7_pseudotime_by_type")


def fig_paired(adata, tag, cluster_key):
    obs = adata.obs.copy()
    obs["barcode"] = obs.index.str.rsplit("_", n=1).str[0]
    obs["row"] = np.arange(adata.n_obs)
    pr = obs[obs["paired"].astype(bool)]
    r = pr[pr["condition"] == "RNA"].set_index("barcode")
    a = pr[pr["condition"] == "ACT"].set_index("barcode")
    common = r.index.intersection(a.index)
    if len(common) == 0:
        return
    r, a = r.loc[common], a.loc[common]

    z = np.asarray(adata.obsm["X_vitae"])
    d = cdist(z[r["row"].values], z[a["row"].values])
    td = np.diag(d)
    fos = ((d < td[:, None]).mean(1) + (d < td[None, :]).mean(0)) / 2

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.8))
    ok = r["pseudotime"].notna().values & a["pseudotime"].notna().values
    x, y = r["pseudotime"].values[ok], a["pseudotime"].values[ok]
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    hb = a1.hexbin(x, y, gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2, edgecolors="white")
    lim = [0, np.nanmax(np.r_[x, y])]
    a1.plot(lim, lim, ls="--", lw=1, color="#888888")
    a1.set_xlabel("Pseudotime (RNA)")
    a1.set_ylabel("Pseudotime (ACT)")
    a1.set_title(f"Same cell, two modalities: ρ = {rho:.2f} (n = {ok.sum()})", fontsize=9)
    fig.colorbar(hb, ax=a1, label="n pairings")

    a2.hist(fos, bins=40, color="#555555", edgecolor="white", linewidth=0.5)
    a2.axvline(0.5, ls="--", lw=1, color="#888888")
    a2.text(0.5, a2.get_ylim()[1] * 0.95, " casuale", color="#666666", fontsize=8, va="top")
    a2.set_xlabel("FOSCTTM for each pair")
    a2.set_ylabel("n pairs")
    a2.set_title(f"Latent space alignment: mean = {fos.mean():.3f}", fontsize=9)

    by = (pd.DataFrame({"ct": r[cluster_key].astype(str).values, "f": fos})
          .groupby("ct")["f"].agg(["mean", "size"]).sort_values("mean"))
    a3.barh(by.index, by["mean"], color="#555555", height=0.6)
    a3.axvline(0.5, ls="--", lw=1, color="#888888")
    for i, (m, n) in enumerate(zip(by["mean"], by["size"])):
        a3.text(m, i, f" {m:.2f} (n={n})", va="center", fontsize=7, color="#333333")
    a3.set_xlabel("FOSCTTM mean")
    a3.set_title("Aligned according to celltype", fontsize=9)
    a3.set_xlim(0, max(0.55, by["mean"].max() * 1.35))
    fig.tight_layout()
    _save(fig, f"{tag}_fig8_paired")


def make_figures(adata, tag, cluster_key="cell_type_abbr", markers=None):
    adata.obs["condition"] = adata.obs["condition"].astype("category")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    fig_latent(adata, tag, cluster_key)
    fig_trajectory(adata, tag, cluster_key)
    summ = fig_likelihood_umap(adata, tag)
    for c in summ:
        adata.obs[c] = summ[c]
    fig_likelihood_vertices(adata, tag)
    fig_likelihood_heatmap(adata, tag, cluster_key)
    fig_likelihood_cells(adata, tag, cluster_key)
    fig_pseudotime_by_type(adata, tag, cluster_key)
    fig_paired(adata, tag, cluster_key)

DEFAULT_MARKERS = ["Pax6", "Sox2", "Hes5", "Eomes", "Neurod2", "Neurod6",
                   "Tbr1", "Bcl11b", "Satb2", "Cux2", "Aqp4", "Olig2"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--cluster_key", default="cell_type_abbr")
    p.add_argument("--markers", nargs="*", default=DEFAULT_MARKERS)
    a = p.parse_args()
    ad_ = sc.read_h5ad(os.path.join(DATA_DIR, f"vitae_results_{a.tag}.h5ad"))
    make_figures(ad_, a.tag, a.cluster_key, a.markers)

