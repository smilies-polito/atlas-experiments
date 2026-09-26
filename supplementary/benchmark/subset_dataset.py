import os
import atlas
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from muon import MuData
from anndata import AnnData

if __name__=="__main__":
    seed = 42
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pct", type=float)
    args = parser.parse_args()

    output_path = os.path.join(os.getcwd(), "output", "benchmark"))
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    data = sc.read_h5ad(os.path.join(os.getcwd(), "data", "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"))
    rna = data[:, data.var["feature_types"]=="GEX"].copy()
    activity = AnnData(X = data.obsm["ATAC_gene_activity"],
                            obs = pd.DataFrame([], index = data.obs_names),
                            var = pd.DataFrame([], index =data.uns["ATAC_gene_activity_var_names"])
    )
    data = MuData({"rna": rna, "activity": activity})
    data.pull_obs()

    n_cells = int(len(data) * args.pct)
    selected_cells = rng.choice(data.obs_names, n_cells, replace = False)
    data = data[data.obs_names.isin(selected_cells), :].copy()

    sc.pp.normalize_total(data["activity"])
    sc.pp.log1p(data["rna"])
    sc.pp.pca(data["activity"], random_state = seed)
    sc.pp.pca(data["rna"], random_state = seed)
    data["rna"].obsm["X_pca"] = data["rna"].obsm["GEX_X_pca"]

    knn = {"rna": 15, "activity":15}
    n_pcs = {"rna": 20, "activity":10}
    use_rep = "X_pca"
    n_neighbors = None

    atlas.pp.wnn(
           mudata=data,
           modalities=("rna", "activity"),
           knn=knn,
           n_pcs=n_pcs,
           use_rep=use_rep,
           n_neighbors=n_neighbors,
           random_state=seed,
    )

    data.write(os.path.join(output_path, f"data_{args.pct}.h5mu"))
    
    

