import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import pickle
import networkx as nx
import anndata as ad
from anndata import AnnData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import VITAE
from VITAE.utils import reset_random_seeds, _comp_dist
from scipy.spatial.distance import cdist
from vitae_figures import make_figures, DEFAULT_MARKERS

WD = os.getcwd()
DATA_DIR = os.path.join(WD, "output", "embryonic_mouse_brain", "vitae_input")
FIG_DIR = os.path.join(WD, "figures")
SEED = 42


def _split_cells(obs_names, labels, paired_frac=0.3, seed=SEED):
    rng = np.random.default_rng(seed)
    obs_names = np.asarray(obs_names)
    labels = np.asarray(labels).astype(str)   
    rna_only, act_only, paired = [], [], []
    for ct in np.unique(labels):
        idx = rng.permutation(obs_names[labels == ct])
        n_p = int(round(len(idx) * paired_frac))
        rest = idx[n_p:]
        paired.extend(idx[:n_p])
        rna_only.extend(rest[: len(rest) // 2])
        act_only.extend(rest[len(rest) // 2:])
    return rna_only + paired, act_only + paired, paired


def _load_mod(code: str, mod: str) -> AnnData:
    d = os.path.join(DATA_DIR)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} does not exist")

    X = sparse.load_npz(os.path.join(d, f"{mod}_X.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, f"{mod}_obs.csv"), index_col=0)
    var = pd.read_csv(os.path.join(d, f"{mod}_var.csv"), index_col=0)
    obs.index = obs.index.astype(str)
    var = pd.DataFrame(index=var.index.astype(str))
    if X.shape != (len(obs), len(var)):
        raise ValueError(f"{mod}: shape of X {X.shape} mismatches with obs/var ({len(obs)}, {len(var)})")
    for c in obs.select_dtypes(include="object").columns:   
        obs[c] = obs[c].astype("category")
    return AnnData(X=X, obs=obs, var=var)


def _prepare_data(code: str, tag: str, cluster_key: str,
                  paired_frac: float = 0.3, seed: int = SEED) -> AnnData:
    path = os.path.join(DATA_DIR, f"vitae_{tag}.h5ad")
    if os.path.exists(path):
        return sc.read_h5ad(path)

    rna = _load_mod(code, "rna")
    act = _load_mod(code, "activity")
    if cluster_key not in rna.obs.columns:
        raise KeyError(f"'{cluster_key}' not in rna.obs. Columns: {list(rna.obs.columns)}")

    genes = rna.var_names.intersection(act.var_names)
    rna, act = rna[:, genes].copy(), act[:, genes].copy()

    cells = rna.obs_names.intersection(act.obs_names)
    labels = rna.obs.loc[cells, cluster_key].values
    rna_cells, act_cells, paired = _split_cells(cells, labels, paired_frac, seed)
    rna, act = rna[rna_cells].copy(), act[act_cells].copy()

    sc.pp.log1p(act)

    hvg = set()
    for a in (rna, act):
        sc.pp.highly_variable_genes(a)
        hvg |= set(a.var_names[a.var["highly_variable"]])
    hvg = sorted(hvg)
    rna, act = rna[:, hvg].copy(), act[:, hvg].copy()

    sc.pp.scale(rna, max_value=10)
    sc.pp.scale(act, max_value=10)

    rna.obs_names = rna.obs_names + "_RNA"
    act.obs_names = act.obs_names + "_ACT"
    adata = ad.concat([rna, act], label="condition", keys=["RNA", "ACT"])
    adata.obs["paired"] = adata.obs_names.str.rsplit("_", n=1).str[0].isin(paired)
    adata.uns["paired_barcodes"] = list(paired)

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    adata.write(path)
    return adata


def _savefig(name: str):
    plt.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close("all")


def _run_vitae(adata: AnnData, tag: str, cluster_key: str, root: str,
               seed: int = SEED):
    os.makedirs(FIG_DIR, exist_ok=True)
    reset_random_seeds(seed)
    tf.keras.backend.clear_session()

    model = VITAE.VITAE(
        adata=adata,
        covariates=["condition"],
        conditions="condition",
        model_type="Gaussian",
        npc=min(64, adata.n_vars - 1),
        hidden_layers=[32, 16],
        latent_space_dim=8,
    )

    ckpt = os.path.join(DATA_DIR, f"vitae_pretrain_{tag}.checkpoint")
    if os.path.exists(ckpt + ".index"):
        model.load_model(path_to_file=ckpt)
    else:
        model.pre_train(gamma=.6, phi=.6,
                        early_stopping_tolerance=0.01, early_stopping_relative=True)
        model.save_model(path_to_file=ckpt)

    model.visualize_latent(color=[cluster_key, "condition"], method="UMAP")
    _savefig(f"{tag}_pretrain_latent.png")

    dist = _comp_dist(model._adata_z.obsm["X_umap"], model.adata.obs[cluster_key].values)
    model.init_latent_space(cluster_label=cluster_key, ratio_prune=0.8, topk=2, dist=dist)

    names = model.labels_map["label_names"].values
    sns.heatmap(model.pi, cmap="viridis", xticklabels=names, yticklabels=names,
                mask=(model.pi == 0))
    _savefig(f"{tag}_pi.png")

    model.train(gamma=1., phi=1.,
                early_stopping_tolerance=0.01, early_stopping_relative=True)
    model.posterior_estimation()

    model.visualize_latent(color=["vitae_new_clustering", cluster_key, "condition"],
                           method="UMAP")
    _savefig(f"{tag}_latent.png")

    model.infer_backbone(method="raw_map")
    _savefig(f"{tag}_backbone_vitae_clusters.png")
    model.plot_backbone(color=cluster_key)
    _savefig(f"{tag}_backbone.png")

    model.infer_trajectory(root, size=20, cmap="viridis")
    _savefig(f"{tag}_trajectory.png")

    _save_outputs(model, tag)   

    try:
        model.plot_center(color="vitae_new_clustering", add_egde=True, add_direct=True)
        _savefig(f"{tag}_centers.png")
    except Exception as e:
        plt.close("all")
        print(f"plot_center failed wth exception: ({e}). Continuing..")
    return model


def _save_outputs(model, tag: str):

    a = model.adata
    names = model.labels_map["label_names"].values
    a.obsm["X_vitae"] = np.asarray(model.z)
    a.obsm["X_vitae_umap"] = model._adata_z.obsm["X_umap"]
    a.obsm["vitae_w_posterior"] = np.asarray(model.cell_position_posterior)
    a.obsm["vitae_w_projected"] = np.asarray(model.cell_position_projected)
    a.obsm["vitae_w_variance"] = np.asarray(model.cell_position_variance)
    a.uns["vitae_vertices"] = list(names)
    a.obsm["vitae_pc_x"] = np.asarray(model.pc_x)
    C = model.inferer.C
    states = [None] * int(C.max() + 1)
    for i in range(len(names)):
        for j in range(i, len(names)):
            states[C[i, j]] = names[i] if i == j else f"{names[i]} — {names[j]}"
    a.uns["vitae_states"] = states
    a.uns["vitae_mu"] = np.asarray(model.mu)
    a.uns["vitae_pi"] = np.asarray(model.pi)

    def _edges(g):
        return pd.DataFrame([(names[u], names[v], float(d.get("weight", np.nan)))
                             for u, v, d in g.edges(data=True)],
                            columns=["from", "to", "score"])
    a.uns["vitae_backbone_edges"] = _edges(model.backbone)
    a.uns["vitae_directed_edges"] = _edges(model.directed_backbone)

    a.write(os.path.join(DATA_DIR, f"vitae_results_{tag}.h5ad"))
    with open(os.path.join(DATA_DIR, f"vitae_graphs_{tag}.pkl"), "wb") as f:
        pickle.dump({"backbone": nx.relabel_nodes(model.backbone, dict(enumerate(names))),
                     "directed": nx.relabel_nodes(model.directed_backbone, dict(enumerate(names)))},
                    f)
    model.save_model(path_to_file=os.path.join(DATA_DIR, f"vitae_trained_{tag}.checkpoint"))


def _evaluate(model, tag: str):

    obs = model.adata.obs.copy()
    obs["barcode"] = obs.index.str.rsplit("_", n=1).str[0]
    obs["row"] = np.arange(len(obs))
    pr = obs[obs["paired"].astype(bool)]
    rna = pr[pr["condition"] == "RNA"].set_index("barcode")
    act = pr[pr["condition"] == "ACT"].set_index("barcode")
    common = rna.index.intersection(act.index)
    rna, act = rna.loc[common], act.loc[common]
    out = {"n_pairs": len(common)}

    z = model.adata.obsm["X_vitae"]
    d = cdist(z[rna["row"].values], z[act["row"].values])
    true_d = np.diag(d)
    out["foscttm"] = float(np.mean([
        ((d[i] < true_d[i]).mean() + (d[:, i] < true_d[i]).mean()) / 2
        for i in range(len(common))
    ]))

    out["same_vitae_cluster"] = float(
        (rna["vitae_new_clustering"].astype(str).values
         == act["vitae_new_clustering"].astype(str).values).mean())

    ok = rna["pseudotime"].notna().values & act["pseudotime"].notna().values
    out["pseudotime_spearman"] = float(pd.Series(rna["pseudotime"].values[ok]).corr(
        pd.Series(act["pseudotime"].values[ok]), method="spearman"))
    out["pseudotime_nan_frac"] = float(obs["pseudotime"].isna().mean())

    print(pd.Series(out).to_string())
    pd.Series(out, name=tag).to_csv(os.path.join(DATA_DIR, f"vitae_metrics_{tag}.csv"))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--knn_rna", type=int, default=15)
    parser.add_argument("--knn_act", type=int, default=15)
    parser.add_argument("--wnn", default=None)
    parser.add_argument("--cluster_key", type=str, default="cell_type_abbr")
    parser.add_argument("--root", type=str, default="IPC")
    parser.add_argument("--paired_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--markers", nargs="*", default=DEFAULT_MARKERS)
    parser.add_argument("--plots_only", action="store_true",
                        help="non ri-traina: rifà le figure dal risultato salvato")
    args = parser.parse_args()
    np.random.seed(args.seed)

    code = f"{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}"
    tag = f"{code}_pf{args.paired_frac}_s{args.seed}"

    if args.plots_only:
        res = sc.read_h5ad(os.path.join(DATA_DIR, f"vitae_results_{tag}.h5ad"))
        make_figures(res, tag, args.cluster_key, args.markers)
        raise SystemExit

    adata = _prepare_data(code, tag, args.cluster_key, args.paired_frac, args.seed)
    labels = set(adata.obs[args.cluster_key].astype(str))
    if args.root not in labels:
        raise ValueError(f"root '{args.root}' not in {sorted(labels)}")

    model = _run_vitae(adata, tag, args.cluster_key, args.root, args.seed)
    _evaluate(model, tag)
    make_figures(model.adata, tag, args.cluster_key, args.markers)
