import os
import atlas
import muon as mu
import numpy as np 
import pandas as pd
import scanpy as sc
import seaborn as sns
import muon.atac as ac
import matplotlib.pyplot as plt
from muon import MuData
from anndata import AnnData
from scipy.io import mmread
from pybiomart import Dataset
from scipy.sparse import load_npz, csr_matrix
from scipy.stats import median_abs_deviation

def anndata_to_frag_bc(x):
    """This function turns atac barcodes into the same format as in the fragment file"""
    parts = str(x).strip().split(".")
    if len(parts) != 8:
        return None
    return f"{parts[0]}.{parts[1]},{parts[2]}.{parts[3]},{parts[4]}.{parts[5]},{parts[6]}.{parts[7]}"

def _to_ucsc(chromosome: str) -> str:
    """This function transforms chromosome into UCSC format"""
    if chromosome == "MT":
        return "chrM"
    else:
        return f"chr{chromosome}"

def _compute_outlier(adata: AnnData, 
                metric: str,
                nmads: int):
    M = adata.obs[metric]
    outlier = (
                        (M < np.median(M) - nmads * median_abs_deviation(M)) | (
                         np.median(M) + nmads * median_abs_deviation(M) < M)
                )
    return outlier


if __name__=="__main__":
    seed = 42
    working_dir = os.getcwd()
    np.random.seed(seed)

    data_path = os.path.join(working_dir, "data", "mouse_skin")
    output_path = os.path.join(working_dir, "output", "mouse_skin")
    rna_matrix = os.path.join(data_path, "GSM4156608_skin.late.anagen.rna.counts.txt")
    atac_matrix = os.path.join(data_path, "GSM4156597_skin.late.anagen.counts.txt")
    barcodes = os.path.join(data_path, "GSM4156597_skin.late.anagen.barcodes.txt") 
    peaks= os.path.join(data_path, "GSM4156597_skin.late.anagen.peaks.bed")
    feature_path = os.path.join(output_path, "features.tsv")
    annotation_path = os.path.join(data_path, "GSM4156597_skin_celltype.txt")
    fragment_file_path = os.path.join(data_path, "GSM4156597_skin.late.anagen.atac.sorted.fragments.bed.gz")

    valid_chr = [f"{i}" for i in range(1,23)] + ["X","MT"]

    barcodes = pd.read_csv(barcodes, sep="\t", index_col=False, header=None)
    barcodes = list(barcodes[0])
    
    clusters_of_interest = ["IRS", "TAC-1", "TAC-2", "Medulla", "Hair-Shaft-cuticle.cortex"]
    annotations = pd.read_csv(annotation_path, sep="\t", index_col=False, header=0) 
    annotations = annotations[annotations["celltype"].isin(clusters_of_interest)]
    rna_to_atac = dict(zip(annotations["rna.bc"], annotations["atac.bc"]))
    atac_to_rna = dict(zip(annotations["atac.bc"], annotations["rna.bc"]))

    rna = pd.read_csv(rna_matrix, header=0, index_col=0, sep= "\t")
    rna = rna.T
    rna = AnnData(X = csr_matrix(rna.values),
                    var = pd.DataFrame([], index = rna.columns),
                    obs = pd.DataFrame([], index = rna.index.values)
    )
    rna.obs_names = rna.obs_names.str.replace(",", ".")
    rna = rna[rna.obs_names.isin(annotations["rna.bc"])].copy() 

    atac =  mmread(atac_matrix)
    atac = csr_matrix(atac)
    atac = atac.T
    peaks = pd.read_csv(peaks, sep="\t", index_col=False, header=None)
    peaks.columns = ["chr", "start", "end"]
    peaks["peak"] = peaks["chr"] + ":" + peaks["start"].astype(str) + "-" + peaks["end"].astype(str)
    atac = AnnData(X = atac, obs = pd.DataFrame([], index=barcodes), var = pd.DataFrame([], index = peaks["peak"]))
    atac = atac[atac.obs_names.isin(annotations["rna.bc"])].copy()

    atac.obs["frag_barcode_raw"] = [rna_to_atac.get(x, None) for x in atac.obs_names]
    atac.obs["frag_barcode_raw"] = atac.obs["frag_barcode_raw"].apply(anndata_to_frag_bc)
    atac.obs_names = atac.obs["frag_barcode_raw"].values

    ac.tl.locate_fragments(atac, fragment_file_path)
    
    ## RNA FEATURES WITH PYBIOMART mm10
    genes = rna.var_names.tolist()
    dataset = Dataset(
                name='mmusculus_gene_ensembl',
                host="http://nov2020.archive.ensembl.org",
            )
    res = dataset.query(attributes=[
            'external_gene_name',
            'chromosome_name',
            'start_position',
            'end_position',
            'strand'
    ])
    res.columns = ["Name", "OR_Chromosome", "Start", "End", "Strand"] 
    features = res[res['Name'].isin(genes) & res["OR_Chromosome"].isin(valid_chr)]
    features["Chromosome"] = features["OR_Chromosome"].apply(_to_ucsc)

    features_collapsed = (
        features
        .groupby("Name", as_index=False)
        .agg({
            "OR_Chromosome": "first",
            "Chromosome": "first",
            "Strand": "first",
             "Start": "min",
            "End": "max",
        })
    )
    

    features_collapsed = features_collapsed.set_index("Name")
    features_collapsed[["Chromosome", "Start", "End", "Strand"]].to_csv(feature_path, sep="\t", header=True, index = True)
    rna = rna[:, rna.var_names.isin(features_collapsed.index)].copy()

    rna.var["mt"] = rna.var_names.str.startswith("mt-")
    rna.var["ribo"] = rna.var_names.str.startswith(("rps", "rpl"))
    sc.pp.calculate_qc_metrics(rna, qc_vars = ["mt", "ribo"], inplace=True, log1p = True)

    sc.pl.violin(rna, 
                ["total_counts", "pct_counts_mt", "n_genes_by_counts"],
                multi_panel = True, save = "SKIN_rna_violin.png")
    sc.pl.scatter(rna, 
                    x = "total_counts",
                    y = "n_genes_by_counts",
                    color = "pct_counts_mt",
                    show = True,
                    save = "SKIN_rna_QC.png")
    rna.obs["outlier"] = ( _compute_outlier(rna, "log1p_total_counts", 5) |
                            _compute_outlier(rna, "log1p_n_genes_by_counts", 5) |
                            (rna.obs["pct_counts_mt"] > 10)
                        )

    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace= True)
    ac.tl.nucleosome_signal(atac, n=1e6)
    nuc_threshold = 2
    atac.obs["nuc_filter"] = ["NUC_FAIL" if ns > nuc_threshold else "NUC_PASS" for ns in atac.obs["nucleosome_signal"] ]
    fig, axs = plt.subplots(figsize=(7, 3.5))
    sns.histplot(atac.obs, x="nucleosome_signal", ax=axs)
    plt.savefig(os.path.join(os.getcwd(), "figures", "SKIN_nuc.png"))
    plt.close()

    atac.obs["outlier"] = atac.obs["nuc_filter"] == "NUC_FAIL"

    features_collapsed = features_collapsed[features_collapsed["Chromosome"] != "chrM"] # chrM not present in fragments 
    activity = ac.tl.count_fragments_features(atac, features = features_collapsed, stranded=True, count_reads=False)
    activity.obs["outlier"] = atac.obs["outlier"]
    activity.obs_names = activity.obs_names.str.replace(",", ".")
    activity.obs_names = [atac_to_rna.get(x, None) for x in activity.obs_names]


    # Filtering 
    data = MuData({"rna":rna, "activity":activity})
    mask = ~(data["rna"].obs["outlier"] | data["activity"].obs["outlier"])
    data = data[mask, :].copy()
   
    annotations = annotations.set_index("rna.bc")
    data.obs = data.obs.join(annotations)

    # RNA-seq preprocessing
    sc.pp.normalize_total(data["rna"])
    sc.pp.normalize_total(data["activity"])
    sc.pp.log1p(data["rna"])
    sc.pp.highly_variable_genes(data["rna"], n_top_genes = 2000)
    sc.pp.pca(data["rna"], random_state=seed)
    sc.pp.pca(data["activity"], random_state=seed)
    sc.pl.pca_variance_ratio(data["rna"], save = "SKIN_rna_pca.png")     
    sc.pl.pca_variance_ratio(data["activity"], save = "SKIN_activity_pca.png")     

    data.write(os.path.join(output_path, "skin.h5mu"))




    
        
                        
    
     


                
