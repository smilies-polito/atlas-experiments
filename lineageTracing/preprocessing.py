import os, gc
import muon as mu
import scanpy as sc
import scipy
import numpy as np
import anndata
import pandas as pd
from anndata import AnnData
from muon import MuData
from muon import atac as ac
from scipy.sparse import csr_matrix

def create_feature_map(features: pd.DataFrame, strand:bool = False) -> pd.DataFrame:
	columns = ["Chromosome", "Start", "End"]
	if strand:
		columns.append("Strand")
	features = features[columns]
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 
	return features


def process_single_atac_experiment(experiment: AnnData, fragment_path:str, features: pd.DataFrame) -> AnnData:
	mu.atac.tl.locate_file(data=experiment, file=fragment_path, key="fragments")
	ac.tl.nucleosome_signal(experiment)
	ac.tl.tss_enrichment(experiment, features)
	return experiment

def compute_single_activity_experiment(experiment:AnnData, fragment_path: str, features: pd.DataFrame, stranded:bool=False) -> AnnData:
	mu.atac.tl.locate_file(data=experiment, file=fragment_path, key ="fragments")
	activity = mu.atac.tl.count_fragments_features(data=experiment, features=features, stranded=stranded)
	return activity 
	

if __name__=="__main__":
	seed = 42
	donor =  ...
	data_path = os.path.join(os.getcwd(), donor)
	metadata = pd.read_csv(..., header=0, index_col=0, sep=",")
	barcodes = pd.read_csv(..., header=0, index_col=False).values.flatten()

	rna_features = pd.read_csv(..., header=None, index_col=False).values.flatten()
	
	rna = scipy.io.mmread( ... )
	rna = AnnData(X=csr_matrix(rna), obs = pd.DataFrame(data=None, columns=None, index=barcodes), var= pd.DataFrame(data=None, columns=None, index= rna_features))
	
	# RNA quality control and filtering
	sc.pp.calculate_qc_metrics(rna, inplace=True)
	sc.pl.violin(rna, ["n_genes_by_counts", "total_counts"], show=False, save="rna_qc.png")
	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=15000)
	
	# RNA extract features from h5 files
	bmmc_data = sc.read_10x_h5( ... )
	bmmc_data.var_names_make_unique()
	hspc_data = sc.read_10x_h5( ... )
	hspc_data.var_names_make_unique()
	hsc_data = sc.read_10x_h5( ... )
	hsc_data.var_names_make_unique()
	vars_complete = bmmc_data.var.combine_first(hspc_data.var).combine_first(hsc_data.var)
	vars_complete[['Chromosome', 'Start', 'End']] = vars_complete['interval'].str.extract(r'([^:]+):(\d+)-(\d+)')
	features = create_feature_map(vars_complete, strand=False)
	
	# ATAC anndata creation 
	atac_features = pd.read_csv(..., header=None, index_col=False).values.flatten()
	atac = scipy.io.mmread( ... )
	atac = AnnData(X=csr_matrix(atac), obs = pd.DataFrame(data=None, columns=None, index=barcodes), var= pd.DataFrame(data=None, columns=None, index= atac_features))

	# ATAC preprocessing + COMPUTE ACTIVITY: devo dividere per tipologia perchè i fragment files sono distinti in base all'esperimento (experimento = bmmc, hspc, hsc)
	bmmc = atac[metadata.Sample=="DN4_BMMC"]
	hspc = atac[metadata.Sample=="DN4_HSPC"] # cellule terminano con "-2"
	hspc.obs_names = hspc.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	hsc = atac[metadata.Sample=="DN4_HSC"] # cellule terminano con "-3"
	hsc.obs_names = hsc.obs_names.map(lambda x: x.split("-")[0] + "-1") 

	# ATAC quality control + filtering su tutto il dataset merged
	bmmc_atac = process_single_atac_experiment(bmmc, fragment_path= ..., features= features)
	hspc_atac = process_single_atac_experiment(hspc, fragment_path= ..., features= features)
	hspc_atac.obs_names = hspc_atac.obs_names.map(lambda x: x.split("-")[0] + "-2")
	hsc_atac = process_single_atac_experiment(hsc, fragment_path= ..., features= features)
	hsc_atac.obs_names = hsc_atac.obs_names.map(lambda x: x.split("-")[0] + "-3")

	atac = anndata.concat([bmmc_atac, hspc_atac, hsc_atac], axis=0)	
	sc.pl.violin(atac, ["nucleosome_signal", "tss_score"], save="atac_qc.png", show=False)
	low_tss, high_tss = np.percentile(atac.obs["tss_score"], 5), np.percentile(atac.obs["tss_score"], 95)
	low_nucleosome, high_nucleosome = np.percentile(atac.obs["nucleosome_signal"], 5), np.percentile(atac.obs["nucleosome_signal"], 95)
	conditions = (atac.obs.nucleosome_signal>low_nucleosome) & (atac.obs.nucleosome_signal< high_nucleosome) & (atac.obs.tss_score > low_tss)
	atac = atac[conditions, :]
	
	# ACTIVITY computation + creazione di una matrice di attività
	sub_metadata = metadata.loc[list(atac.obs_names), :]
	bmmc = atac[sub_metadata.Sample=="DN4_BMMC"]
	hspc = atac[sub_metadata.Sample=="DN4_HSPC"] # cellule terminano con "-2"
	hspc.obs_names = hspc.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	hsc = atac[sub_metadata.Sample=="DN4_HSC"] # cellule terminano con "-3"
	hsc.obs_names = hsc.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	
	bmmc_activity = compute_single_activity_experiment(bmmc, fragment_path= ..., features= features)
	hspc_activity = compute_single_activity_experiment(hspc, fragment_path= ..., = features)
	hspc_activity.obs_names = hspc_activity.obs_names.map(lambda x: x.split("-")[0] + "-2")
	hsc_activity = compute_single_activity_experiment(hsc, fragment_path= ..., = features)
	hsc_activity.obs_names = hsc_activity.obs_names.map(lambda x: x.split("-")[0] + "-3")

	activity = anndata.concat([bmmc_activity, hspc_activity, hsc_activity], axis=0)

	# INTERSECTING GENES AND BARCODES
	intersecting_cells = list(set(rna.obs_names).intersection(set(activity.obs_names)))	
	intersecting_genes = list(set(activity.var_names).intersection(set(rna.var_names)))

	activity = activity[activity.obs_names.isin(intersecting_cells), activity.var_names.isin(intersecting_genes)].copy()
	activity = activity[intersecting_cells, intersecting_genes]
	rna = rna[rna.obs_names.isin(intersecting_cells), rna.var_names.isin(intersecting_genes)].copy()
	rna = rna[intersecting_cells, intersecting_genes]
	
	# PREPROCESSING: target sum in rna required for celltypist
	sc.pp.normalize_total(rna, target_sum=1e4)
	sc.pp.log1p(rna)
	sc.pp.highly_variable_genes(rna)
	sc.pp.normalize_total(activity)

	# MUON DATASET 
	data = MuData({"rna": rna, "activity":activity})
	data.obs = data.obs.merge(metadata, left_index=True, right_index=True, how="left")

	# PCA, NEIGHBORS E WNN + MULTIMODAL UMAP 
	sc.pp.pca(data["rna"], random_state=seed)
	sc.pp.pca(data["activity"], random_state=seed)
	sc.pl.pca_variance_ratio(data["rna"], save="variance_ratio_rna.png", show=False)
	sc.pl.pca_variance_ratio(data["activity"], save="variance_ratio_activity.png", show=False)

	n_pcs_rna = 15
	n_pcs_activity = 5
	knn_rna = 20
	knn_activity = 20
	wnn = 20

	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, n_pcs = n_pcs_rna, random_state = seed)
	sc.pp.neighbors(data["activity"], n_neighbors=knn_activity, n_pcs = n_pcs_activity, random_state = seed)
	mu.pp.neighbors(data, key_added="wnn", n_neighbors=wnn, random_state=seed)
	mu.tl.umap(data, random_state = seed, neighbors_key="wnn")
	mu.tl.louvain(data, random_state=seed)
	mu.tl.leiden(data, random_state=seed)
	mu.pl.umap(data, color=["STD.CellType", "louvain", "leiden"], legend_loc="on data", show=False, save="wnn_umap.png")
	
	data.write( ... )
