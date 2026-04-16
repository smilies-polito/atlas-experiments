import os
import atlas
import muon as mu
import numpy as np 
import pandas as pd
import scanpy as sc
import seaborn as sns
import muon.atac as ac
import matplotlib.pyplot as plt
from .utils import _compute_outlier
from muon import MuData


if __name__=="__main__":
    seed = 42
    working_dir = os.getcwd()
    np.random.seed(seed)
    n_pcs_rna, n_pcs_act = 20, 10
    knn_rna, knn_act, wnn = 20,20,20

    data_path = os.path.join(working_dir, "data", "embryonic_mouse_brain")
    output_path = os.path.join(working_dir, "output", "embryonic_mouse_brain")
    ffbcm_path = os.path.join(data_path, "filtered_feature_bc_matrix")
    feature_path = os.path.join(output_path, "features.tsv")
    annotation_path = os.path.join(data_path, "cell_annotations.tsv")
    fragment_file_path = os.path.join(data_path, "e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz")

    valid_chr = [f"chr{i}" for i in range(1,23)] + ["chrX","chrY","chrM"]
    gene_metadata = pd.read_csv(os.path.join(ffbcm_path, "features.tsv.gz"), sep="\t", header=None)
    gene_metadata.columns = ["id", "symbol", "type", "Chromosome", "Start", "End"]
    gene_metadata = gene_metadata[gene_metadata["type"] == "Gene Expression"]

    data = sc.read_10x_mtx(ffbcm_path, 
                            var_names = "gene_symbols",
                            gex_only = False)

    rna = data[:, data.var["feature_types"]=="Gene Expression"].copy()    
    rna.var = pd.merge(rna.var, gene_metadata, left_on="gene_ids", right_on = "id", how="left").drop(["id", "type"], axis=1).set_index("symbol")
    rna.var_names_make_unique()
    features = rna[:, rna.var["Chromosome"].isin(valid_chr)].var[["Chromosome", "Start", "End"]]
    features.to_csv(feature_path, sep="\t", index =True)

    atac = data[:, ~(data.var["feature_types"]=="Gene Expression")].copy()    
    ac.tl.locate_fragments(atac, fragment_file_path)

    # rna preprocessing
    rna.var["mt"] = rna.var_names.str.startswith("mt-")
    rna.var["ribo"] = rna.var_names.str.startswith(("rps", "rpl"))
    sc.pp.calculate_qc_metrics(rna, qc_vars = ["mt", "ribo"], inplace=True, log1p = True)

    sc.pl.violin(rna, 
                ["total_counts", "pct_counts_mt", "n_genes_by_counts"],
                multi_panel = True, save = "EMB_rna_violin.png")
    sc.pl.scatter(rna, 
                    x = "total_counts",
                    y = "n_genes_by_counts",
                    color = "pct_counts_mt",
                    show = True,
                    save = "EMB_rna_QC.png")
    rna.obs["outlier"] = ( _compute_outlier(rna, "log1p_total_counts", 5) |
                            _compute_outlier(rna, "log1p_n_genes_by_counts", 5) |
                            (rna.obs["pct_counts_mt"] > 10)
                        )
    # activity 
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace= True)
    ac.tl.nucleosome_signal(atac, n=1e6)
    nuc_threshold = 2
    atac.obs["nuc_filter"] = ["NUC_FAIL" if ns > nuc_threshold else "NUC_PASS" for ns in atac.obs["nucleosome_signal"] ]
    fig, axs = plt.subplots(figsize=(7, 3.5))
    sns.histplot(atac.obs, x="nucleosome_signal", ax=axs)
    plt.savefig(os.path.join(os.getcwd(), "figures", "EMB_nuc.png"))
    plt.close()


    tss_enr = ac.tl.tss_enrichment(atac, features= features, random_state = seed)
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
    p1 = sns.histplot(atac.obs, x="tss_score", ax=axs[0])
    p1.set_title("Full range")
    p2 = sns.histplot(
                    atac.obs,
                    x="tss_score",
                    binrange=(0, atac.obs["tss_score"].quantile(0.995)),
                    ax=axs[1],
        )
    p2.set_title("Up to 99.5% percentile")
    plt.savefig(os.path.join(os.getcwd(), "figures", "EMB_histplot_tss.png"))
    plt.close()

    tss_threshold = 1.5
    tss_enr.obs["tss_filter"] = ["TSS_FAIL" if score < tss_threshold else "TSS_PASS" for score in atac.obs["tss_score"] ]
    atac.obs["tss_filter"] = ["TSS_FAIL" if score < tss_threshold else "TSS_PASS" for score in atac.obs["tss_score"] ]

    fig, ax = plt.subplots()
    ac.pl.tss_enrichment(tss_enr, color="tss_filter", ax = ax)
    plt.savefig(os.path.join(os.getcwd(), "figures", "EMB_tss_filter.png"))
    plt.close()
    
    sc.pl.scatter(atac, 
                    x = "total_counts",
                    y = "n_genes_by_counts",
                    color = "tss_score",
                    show = True,
                    save = "EMB_atac_qc.png")
    plot_tss_max = 20
    g = sns.jointplot(data=atac[(atac.obs["tss_score"] < plot_tss_max)].obs,
                        x="total_counts",
                        y="tss_score",
                        color="black",
                        marker=".",
        )
    # Density plot including lines
    g.plot_joint(sns.kdeplot, fill=True, cmap="Blues", zorder=1, alpha=0.75)
    g.plot_joint(sns.kdeplot, color="black", zorder=2, alpha=0.75)
    plt.savefig(os.path.join(os.getcwd(), "figures", "EMB_kde.png"))
    plt.close()

    atac.obs["outlier"] = ((atac.obs["tss_filter"] == "TSS_FAIL") |
                            (atac.obs["tss_score"] > 15) | (atac.obs["nuc_filter"] == "NUC_FAIL"))

    # Filtering 
    data = MuData({"rna":rna, "atac":atac})
    mask = ~(data.obs["rna:outlier"] | data.obs["atac:outlier"])
    data = data[mask, :].copy()
    
    annotations = pd.read_csv(annotation_path, sep="\t", header=0, index_col=0)
    data.obs = data.obs.join(annotations)

    non_developing_clusters = ["Cajal-Retzius", "Interneurons1", "Interneurons2", "Interneurons3", "Microglia", np.nan]
    data = data[~data.obs["celltype"].isin(non_developing_clusters)].copy()

    # RNA-seq preprocessing
    sc.pp.normalize_total(data["rna"])
    sc.pp.log1p(data["rna"])
    sc.pp.highly_variable_genes(data["rna"], n_top_genes = 2000)
    sc.pp.pca(data["rna"], random_state=seed)
    sc.pl.pca_variance_ratio(data["rna"])     

    data.write(os.path.join(output_path, "emb.h5mu"))




    
        
                        
    
     


                
