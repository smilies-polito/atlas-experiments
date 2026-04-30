import os
import atlas
import argparse
import pybiomart
import numpy as np
import pandas as pd
import muon as mu
import scanpy as sc
from muon import MuData
from anndata import AnnData


if __name__=="__main__":
    seed = 42
    np.random.seed(seed)
    pcs_rna, pcs_act = 20, 10
    knn_rna, knn_act, wnn = 15, 15, None

    parser = argparse.ArgumentParser()
    parser.add_argument("--removed", type=int, default=1)
    args = parser.parse_args()
    removed = bool(args.removed)
    prefix = "brainR" if removed else "brain"

    data_path = os.path.join(os.getcwd(), "data", "human_brain")
    output_path = os.path.join(os.getcwd(), "output", "human_brain")

    clusters = pd.read_csv(os.path.join(data_path, "GSE162170_multiome_cluster_names.txt"), 
                            sep = "\t", header = 0)
    clusters = clusters[clusters["Assay"] == "Multiome RNA"]
    cells_metadata = pd.read_csv(os.path.join(data_path, "GSE162170_multiome_cell_metadata.txt"), 
                            sep = "\t", header = 0)
    cells_metadata = cells_metadata.merge(clusters, how="left", left_on = "seurat_clusters", right_on="Cluster.ID")
    cells_metadata.index = cells_metadata["Sample.ID"] + "_" + cells_metadata["Cell.Barcode"]

    outlier = pd.read_csv(os.path.join(output_path, "to_remove.tsv"), sep = "\t", index_col=0, header=0).index 


    rna = pd.read_csv(os.path.join(data_path, "GSE162170_multiome_rna_counts.tsv.gz"), sep = "\t").T
    activity = pd.read_csv(os.path.join(data_path, "GSE162170_multiome_atac_gene_activities.tsv.gz"), sep = "\t").T

    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    query = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name']) 
    query.set_index("Gene stable ID", inplace=True)
    query = query[~query["Gene name"].isna()]
    intersection = list(set(rna.columns).intersection(set(query.index)))
    query = query.loc[intersection]
    rna = rna[intersection]
    var_names = [query.loc[g, "Gene name"] for g in rna.columns]
    rna.columns = var_names

    activity_cols = [s for s in activity.columns if not s.startswith("MT-")]
    activity = activity[activity_cols]

    rna = AnnData(rna)
    rna.var_names_make_unique()
    activity = AnnData(activity)
    activity.var_names_make_unique()

    sc.pp.normalize_total(rna)
    sc.pp.normalize_total(activity)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=2000)
    sc.pp.pca(rna, random_state = seed)
    sc.pp.pca(activity, random_state = seed)

    sc.pl.pca_variance_ratio(rna, save = "_HF_rna.png")
    sc.pl.pca_variance_ratio(activity, save = "_HF_activity.png")

    data = MuData({"rna": rna, "activity": activity})
    data.obs = data.obs.merge(cells_metadata, how ="left", left_index = True, right_index = True)
    multivelo_mapping = {'RG': 'RG, Astro', 'nIPC/GluN1': 'nIPC, ExN', 'GluN3': 'ExM', 'GluN2': 'ExUp', 'GluN4': 'ExDp', 'GluN5': 'ExDp', "mGPC/OPC": "mGPC, OPC"}
    data.obs["Cluster.Name"] = data.obs["Cluster.Name"].replace(multivelo_mapping)
    developing_clusters = ["SP", "mGPC, OPC", "RG, Astro", "Cyc. Prog.", "ExUp", "ExDp", "ExM", "nIPC, ExN"]
    data = data[data.obs["Cluster.Name"].isin(developing_clusters)].copy()
    data.obs["Cluster.Name"] = data.obs["Cluster.Name"].astype("category")

    data["activity"].obs = data["activity"].obs.merge(data.obs[["Sample.ID", "Cluster.Name"]], how="left", left_index=True, right_index=True)
    data["rna"].obs = data["rna"].obs.merge(data.obs[["Sample.ID", "Cluster.Name"]], how="left", left_index=True, right_index=True)

    if removed:
        outlier = pd.read_csv(os.path.join(data_path, "to_remove.tsv"), sep = "\t", index_col=0, header=0).index 
        data = data[~data.obs_names.isin(outlier)].copy()

    sc.pp.neighbors(data["rna"], n_pcs = pcs_rna, n_neighbors=knn_rna, random_state = seed)
    sc.pp.neighbors(data["activity"], n_pcs = pcs_act, n_neighbors=knn_act, random_state = seed)
    mu.pp.neighbors(data, n_neighbors = wnn, random_state = seed, key_added = "wnn")
    mu.tl.umap(data, random_state = seed, neighbors_key = "wnn")
    mu.pl.embedding(data, basis="X_umap", color = ["Cluster.Name"], save = "_HFR.png")

    data.write(os.path.join(output_path, f"{prefix}.h5mu"))




