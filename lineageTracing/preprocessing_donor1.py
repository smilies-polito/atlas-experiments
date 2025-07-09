import os, gc
import muon as mu
import scanpy as sc
import scipy
import numpy as np
import anndata
import pandas as pd
from utils import create_feature_map, process_single_atac_experiment, compute_single_activity_experiment, assign_lineage
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	# Cell cycle genes
	working_dir = ...  #repository path 
	data_path = os.path.join(working_dir, "data", "lineage_tracing", "donor1")
	cc_genes_path = os.path.join(working_dir, "data", "lineage_tracing", "genes_cellcycle.tsv")
	cc_genes = pd.read_csv(cc_genes_path, header=0, index_col=False, sep ="\t")
	
	metadata_t1 = pd.read_csv(os.path.join(data_path, "all.metadata_T1.csv"), header=0, index_col=0) # metadata from figshare 
	metadata_t1["Time"] = "T1"
	barcodes_t1 = pd.read_csv(os.path.join(data_path, "all.barcodes_T1.csv"), header=0, index_col=False).values.flatten() # barcodes from figshare link 
	rna_features_t1 = pd.read_csv(os.path.join(data_path, "all.genes_T1.csv"), header=None, index_col=False).values.flatten() # gene names from figshare link 
	metadata_t2 = pd.read_csv(os.path.join(data_path, "all.metadata_T2.csv"), header=0, index_col=0) # metadata from figshare 
	metadata_t2["Time"] = "T2"
	barcodes_t2 = pd.read_csv(os.path.join(data_path, "all.barcodes_T2.csv"), header=0, index_col=False).values.flatten() # barcodes from figshare link 
	rna_features_t2 = pd.read_csv(os.path.join(data_path, "all.genes_T2.csv"), header=None, index_col=False).values.flatten() # gene names from figshare link 

	rna_t1 = scipy.io.mmread(os.path.join(data_path, "all.rna_T1.mtx")) #rna counts from figshare
	rna_t1 = AnnData(X=csr_matrix(rna_t1), obs = pd.DataFrame(data=None, columns=None, index=barcodes_t1), var= pd.DataFrame(data=None, columns=None, index= rna_features_t1))
	rna_t1.obs_names = rna_t1.obs_names.map(lambda x: x + ":t1")
	rna_t2 = scipy.io.mmread(os.path.join(data_path, "all.rna_T2.mtx")) #rna counts from figshare
	rna_t2 = AnnData(X=csr_matrix(rna_t2), obs = pd.DataFrame(data=None, columns=None, index=barcodes_t2), var= pd.DataFrame(data=None, columns=None, index= rna_features_t2))
	rna_t2.obs_names = rna_t2.obs_names.map(lambda x: x+":t2")

	rna = anndata.concat((rna_t1, rna_t2), axis=0)

	# RNA quality control and filtering
	sc.pp.calculate_qc_metrics(rna, inplace=True)
	sc.pl.violin(rna, ["n_genes_by_counts", "total_counts"])
	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=15000)

	# RNA extract features from h5 files
	bmmc_t1 = sc.read_10x_h5(os.path.join(data_path, "GSE219106_Young1_BMMC.filtered_feature_bc_matrix.h5")) #h5 from GSE219015
	bmmc_t1.var_names_make_unique()
	hspc_t1 = sc.read_10x_h5(os.path.join(data_path, "GSE219106_Young1_HSPC.filtered_feature_bc_matrix.h5")) #h5 from GSE219015
	hspc_t1.var_names_make_unique()
	hsc_t1 = sc.read_10x_h5(os.path.join(data_path, "GSE219106_Young1_HSC.filtered_feature_bc_matrix.h5")) # h5 from GSE219015   
	hsc_t1.var_names_make_unique()
	vars_complete = bmmc_t1.var.combine_first(hspc_t1.var).combine_first(hsc_t1.var)
	vars_complete[['Chromosome', 'Start', 'End']] = vars_complete['interval'].str.extract(r'([^:]+):(\d+)-(\d+)')
	features_t1 = create_feature_map(vars_complete, strand=False)

	bmmc_t2 = sc.read_10x_h5(os.path.join(data_path, "GSE219167_Young1_T2_BMMC.filtered_feature_bc_matrix.h5")) #h5 from GSE219015
	bmmc_t2.var_names_make_unique()
	hspc_t2 = sc.read_10x_h5(os.path.join(data_path, "GSE219167_Young1_T2_HSPC.filtered_feature_bc_matrix.h5")) #h5 from GSE219015
	hspc_t2.var_names_make_unique()
	hsc_t2 = sc.read_10x_h5(os.path.join(data_path, "GSE219167_Young1_T2_HSC.filtered_feature_bc_matrix.h5")) # h5 from GSE219015   
	hsc_t2.var_names_make_unique()
	vars_complete = bmmc_t2.var.combine_first(hspc_t2.var).combine_first(hsc_t2.var)
	vars_complete[['Chromosome', 'Start', 'End']] = vars_complete['interval'].str.extract(r'([^:]+):(\d+)-(\d+)')
	features_t2 = create_feature_map(vars_complete, strand=False)

	# ATAC anndata creation 
	atac_features_t1 = pd.read_csv(os.path.join(data_path, "all.peaks_T1.csv"), header=None, index_col=False).values.flatten() # peaks from figshare
	atac_features_t2 = pd.read_csv(os.path.join(data_path, "all.peaks_T2.csv"), header=None, index_col=False).values.flatten() # peaks from figshare

	atac_t1 = scipy.io.mmread(os.path.join(data_path, "all.atac_T1.mtx")) # atac counts from figshare
	atac_t1 = AnnData(X=csr_matrix(atac_t1), obs = pd.DataFrame(data=None, columns=None, index=barcodes_t1), var= pd.DataFrame(data=None, columns=None, index= atac_features_t1))
	atac_t2 = scipy.io.mmread(os.path.join(data_path, "all.atac_T2.mtx")) # atac counts from figshare
	atac_t2 = AnnData(X=csr_matrix(atac_t2), obs = pd.DataFrame(data=None, columns=None, index=barcodes_t2), var= pd.DataFrame(data=None, columns=None, index= atac_features_t2))
	
	bmmc_t1 = atac_t1[metadata_t1.Sample=="DN4_BMMC"]
	hspc_t1 = atac_t1[metadata_t1.Sample=="DN4_HSPC"] # cellule terminano con "-2"
	hspc_t1.obs_names = hspc_t1.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	hsc_t1 = atac_t1[metadata_t1.Sample=="DN4_HSC"] # cellule terminano con "-3"
	hsc_t1.obs_names = hsc_t1.obs_names.map(lambda x: x.split("-")[0] + "-1") 

	bmmc_atac = process_single_atac_experiment(bmmc_t1, fragment_path= os.path.join(data_path, "GSE219106_Young1_BMMC.atac_fragments.tsv.gz"), features= features_t1) # fragments.tsv from GSE219016  
	hspc_atac = process_single_atac_experiment(hspc_t1, fragment_path= os.path.join(data_path,  "GSE219106_Young1_HSPC.atac_fragments.tsv.gz"), features= features_t1) # fragments.tsv from GSE219016
	hspc_atac.obs_names = hspc_atac.obs_names.map(lambda x: x.split("-")[0] + "-2")
	hsc_atac = process_single_atac_experiment(hsc_t1, fragment_path= os.path.join(data_path, "GSE219106_Young1_HSC.atac_fragments.tsv.gz"), features= features_t1) # fragments.tsv from GSE219016
	hsc_atac.obs_names = hsc_atac.obs_names.map(lambda x: x.split("-")[0] + "-3")
	atac_t1 = anndata.concat([bmmc_atac, hspc_atac, hsc_atac], axis=0)	
	sc.pl.violin(atac_t1, ["nucleosome_signal", "tss_score"])
	low_tss, high_tss = np.percentile(atac_t1.obs["tss_score"], 5), np.percentile(atac_t1.obs["tss_score"], 95)
	low_nucleosome, high_nucleosome = np.percentile(atac_t1.obs["nucleosome_signal"], 5), np.percentile(atac_t1.obs["nucleosome_signal"], 95)
	conditions = (atac_t1.obs.nucleosome_signal>low_nucleosome) & (atac_t1.obs.nucleosome_signal< high_nucleosome) & (atac_t1.obs.tss_score > low_tss)	
	atac_t1 = atac_t1[conditions, :]

	bmmc_t2 = atac_t2[metadata_t2.Sample=="DN4_T2_BMMC"]
	hspc_t2 = atac_t2[metadata_t2.Sample=="DN4_T2_HSPC"] # cellule terminano con "-2"
	hspc_t2.obs_names = hspc_t2.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	hsc_t2 = atac_t2[metadata_t2.Sample=="DN4_T2_HSC"] # cellule terminano con "-3"
	hsc_t2.obs_names = hsc_t2.obs_names.map(lambda x: x.split("-")[0] + "-1") 

	bmmc_atac = process_single_atac_experiment(bmmc_t2, fragment_path= os.path.join(data_path, "GSE219167_Young1_T2_BMMC.atac_fragments.tsv.gz"), features= features_t2) # fragments.tsv from GSE219016  
	hspc_atac = process_single_atac_experiment(hspc_t2, fragment_path= os.path.join(data_path, "GSE219167_Young1_T2_HSPC.atac_fragments.tsv.gz"), features= features_t2) # fragments.tsv from GSE219016
	hspc_atac.obs_names = hspc_atac.obs_names.map(lambda x: x.split("-")[0] + "-2")
	hsc_atac = process_single_atac_experiment(hsc_t2, fragment_path= os.path.join(data_path, "GSE219167_Young1_T2_HSC.atac_fragments.tsv.gz"), features= features_t2) # fragments.tsv from GSE219016
	hsc_atac.obs_names = hsc_atac.obs_names.map(lambda x: x.split("-")[0] + "-3")
	atac_t2 = anndata.concat([bmmc_atac, hspc_atac, hsc_atac], axis=0)	
	sc.pl.violin(atac_t2, ["nucleosome_signal", "tss_score"])
	low_tss, high_tss = np.percentile(atac_t2.obs["tss_score"], 5), np.percentile(atac_t2.obs["tss_score"], 95)
	low_nucleosome, high_nucleosome = np.percentile(atac_t2.obs["nucleosome_signal"], 5), np.percentile(atac_t2.obs["nucleosome_signal"], 95)
	conditions = (atac_t2.obs.nucleosome_signal>low_nucleosome) & (atac_t2.obs.nucleosome_signal< high_nucleosome) & (atac_t2.obs.tss_score > low_tss)	
	atac_t2 = atac_t2[conditions, :]
	
	# ACTIVITY computation + creazione di una matrice di attività
	sub_metadata = metadata_t1.loc[list(atac_t1.obs_names), :]
	bmmc = atac_t1[sub_metadata.Sample=="DN4_BMMC"]
	hspc = atac_t1[sub_metadata.Sample=="DN4_HSPC"] # cellule terminano con "-2"
	hspc.obs_names = hspc.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	hsc = atac_t1[sub_metadata.Sample=="DN4_HSC"] # cellule terminano con "-3"
	hsc.obs_names = hsc.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	
	bmmc_activity = compute_single_activity_experiment(bmmc, fragment_path= os.path.join(data_path, "GSE219106_Young1_BMMC.atac_fragments.tsv.gz"), features= features_t1) # fragments.tsv from GSE219016
	hspc_activity = compute_single_activity_experiment(hspc, fragment_path= os.path.join(data_path, "GSE219106_Young1_HSPC.atac_fragments.tsv.gz"), features= features_t1) # fragments.tsv from GSE219016
	hspc_activity.obs_names = hspc_activity.obs_names.map(lambda x: x.split("-")[0] + "-2")
	hsc_activity = compute_single_activity_experiment(hsc, fragment_path= os.path.join(data_path, "GSE219106_Young1_HSC.atac_fragments.tsv.gz"), features= features_t1) # fragments.tsv from GSE219016
	hsc_activity.obs_names = hsc_activity.obs_names.map(lambda x: x.split("-")[0] + "-3")

	activity_t1 = anndata.concat([bmmc_activity, hspc_activity, hsc_activity], axis=0)
	activity_t1.obs_names = activity_t1.obs_names.map(lambda x: x + ":t1")

	sub_metadata = metadata_t2.loc[list(atac_t2.obs_names), :]
	bmmc = atac_t2[sub_metadata.Sample=="DN4_T2_BMMC"]
	hspc = atac_t2[sub_metadata.Sample=="DN4_T2_HSPC"] # cellule terminano con "-2"
	hspc.obs_names = hspc.obs_names.map(lambda x: x.split("-")[0] + "-1") 
	hsc = atac_t2[sub_metadata.Sample=="DN4_T2_HSC"] # cellule terminano con "-3"
	hsc.obs_names = hsc.obs_names.map(lambda x: x.split("-")[0] + "-1") 

	bmmc_activity = compute_single_activity_experiment(bmmc, fragment_path= os.path.join(data_path, "GSE219167_Young1_T2_BMMC.atac_fragments.tsv.gz"), features= features_t2) # fragments.tsv from GSE219016
	hspc_activity = compute_single_activity_experiment(hspc, fragment_path= os.path.join(data_path, "GSE219167_Young1_T2_HSPC.atac_fragments.tsv.gz"), features= features_t2) # fragments.tsv from GSE219016
	hspc_activity.obs_names = hspc_activity.obs_names.map(lambda x: x.split("-")[0] + "-2")
	hsc_activity = compute_single_activity_experiment(hsc, fragment_path= os.path.join(data_path, "GSE219167_Young1_T2_HSC.atac_fragments.tsv.gz"), features= features_t2) # fragments.tsv from GSE219016
	hsc_activity.obs_names = hsc_activity.obs_names.map(lambda x: x.split("-")[0] + "-3")
	
	activity_t2 = anndata.concat([bmmc_activity, hspc_activity, hsc_activity], axis=0)
	activity_t2.obs_names = activity_t2.obs_names.map(lambda x: x + ":t2")

	activity = anndata.concat((activity_t1, activity_t2), axis=0)

	# INTERSECTING GENES AND BARCODES
	intersecting_cells = list(set(rna.obs_names).intersection(set(activity.obs_names)))	
	intersecting_genes = list(set(activity.var_names).intersection(set(rna.var_names)))

	activity = activity[activity.obs_names.isin(intersecting_cells), activity.var_names.isin(intersecting_genes)].copy()
	activity = activity[intersecting_cells, intersecting_genes]
	rna = rna[rna.obs_names.isin(intersecting_cells), rna.var_names.isin(intersecting_genes)].copy()
	rna = rna[intersecting_cells, intersecting_genes]
	
	# PREPROCESSING: target sum in rna required for celltypist
	rna.var_names = rna.var_names.str.upper()
	sc.pp.normalize_total(rna, target_sum=1e4)
	sc.pp.log1p(rna)
	sc.pp.scale(rna)
	sc.pp.highly_variable_genes(rna)

	s_genes = list(set(cc_genes["G1/S"].values).intersection(set(rna.var_names)))
	g2m_genes = list(set(cc_genes["G2/M"].values).intersection(set(rna.var_names)))
	sc.tl.score_genes_cell_cycle(rna, s_genes = s_genes, g2m_genes =g2m_genes)
	sc.pp.regress_out(rna, ['S_score', 'G2M_score'])
	
	activity.var_names = activity.var_names.str.upper()
	sc.pp.normalize_total(activity)

	# MUON DATASET 
	data = MuData({"rna": rna, "activity":activity})
	
	metadata_t1.index = metadata_t1.index.map(lambda x: x+":t1")
	metadata_t2.index = metadata_t2.index.map(lambda x: x+":t2")
	all_metadata = pd.concat((metadata_t1, metadata_t2))
	data.obs = data.obs.merge(all_metadata, how="left", right_index=True, left_index = True)
	data.obs["lineage"] = data.obs["STD.CellType"].apply(lambda x: assign_lineage(x))

	# PCA, NEIGHBORS E WNN + MULTIMODAL UMAP 
	sc.pp.pca(data["rna"], random_state=seed)
	sc.pp.pca(data["activity"], random_state=seed)

	n_pcs_rna = 15
	n_pcs_activity = 10
	knn_rna = 30
	knn_activity = 30
	wnn = 30
	
	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, n_pcs = n_pcs_rna, random_state = seed)
	sc.pp.neighbors(data["activity"], n_neighbors=knn_activity, n_pcs = n_pcs_activity, random_state = seed)
	mu.pp.neighbors(data, key_added="wnn", n_neighbors=wnn, random_state=seed)
	mu.tl.umap(data, random_state = seed, neighbors_key="wnn")
	mu.pl.umap(data, color=["STD.CellType", "Time", "lineage"], legend_loc="on data", save="muon.png")
	
	data.write(os.path.join(data_path, f"data.h5mu"))
