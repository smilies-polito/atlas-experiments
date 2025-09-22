import os, gc
import muon as mu
import scanpy as sc
import scipy
import numpy as np
import anndata
import pandas as pd
from src.utils import create_feature_map, process_single_atac_experiment, compute_single_activity_experiment, assign_lineage, intermediate_probability, get_state_lineage_tracing, search_cells
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	# Cell cycle genes
	working_dir = "/scvemo"
	data_path = os.path.join(working_dir, "data", "lineage_tracing", "donor1")
	cc_genes_path = os.path.join(working_dir, "data", "genes_cellcycle.tsv")
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
	sc.pl.violin(rna, ["n_genes_by_counts", "total_counts"], save="_rna_donor1_qc.png")
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


 	sc.pp.pca(data["rna"], random_state=seed)
	sc.pp.pca(data["activity"], random_state=seed)
	sc.pl.pca_variance_ratio(data["rna"], show=False, save = "_LT_donor1_rna.png")
	sc.pl.pca_variance_ratio(data["activity"], show=False, save = "_LT_donor1_activity.png")

	n_pcs_rna = 15
	n_pcs_activity = 10
	knn_rna = 30
	knn_activity = 30
	wnn = 30

	#BBKNN
#	sc.external.pp.bbknn(data["rna"], batch_key="Time", use_rep="X_pca", n_pcs=n_pcs_rna)
#	sc.external.pp.bbknn(data["activity"], batch_key="Time", use_rep="X_pca", n_pcs=n_pcs_activity)
#	sc.tl.umap(data["rna"], random_state=seed)
#	sc.tl.umap(data["activity"], random_state=seed)
#	sc.pl.embedding(data["rna"], basis="X_umap", color="Time", save="_rna_LT_donor1_time_bbknn.png")
#	sc.pl.embedding(data["activity"], basis="X_umap", color="Time", save="_activity_LT_donor1_time_bbknn.png")

	# HARMONY
#	data["rna"].obs["Time"] = data["rna"].obs["Time"].astype("category")
#	data["activity"].obs["Time"] = data["activity"].obs["Time"].astype("category")
#	sc.external.pp.harmony_integrate(data["rna"], key="Time", basis="X_pca", adjusted_basis="X_pca_harmony")
#	sc.external.pp.harmony_integrate(data["activity"], key="Time", basis="X_pca", adjusted_basis="X_pca_harmony")
#	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, use_rep="X_pca_harmony", random_state = seed)
#	sc.pp.neighbors(data["activity"], n_neighbors=knn_activity, use_rep="X_pca_harmony", random_state = seed)
#	sc.tl.umap(data["rna"], random_state = seed)
#	sc.tl.umap(data["activity"], random_state = seed)
#	sc.pl.embedding(data["rna"], basis="X_umap", color="Time", save="_rna_donor1_harmony.png")
#	sc.pl.embedding(data["activity"], basis="X_umap", color="Time", save="_activity_donor1_harmony.png")
	

	mu.pp.neighbors(data, key_added="wnn", n_neighbors=wnn, random_state=seed)
	mu.tl.umap(data, random_state = seed, neighbors_key="wnn")
	mu.pl.umap(data, color=["STD.CellType", "Time", "lineage"], legend_loc="on data", save="_LT_donor1_multimodal.png")
	data.write(os.path.join(data_path, f"data.h5mu"))

	# CONSTRUCTING GROUND TRUTH
	threshold = 0.7
	epsilon = 1e-6 
	branch_probabilities_path = os.path.join(data_path, "branch_assignment.csv")
	reachable_lineages = {"CMP": ["erythroid", "myeloid", "megakaryocyte"], 
						  "MPP": ["erythroid", "myeloid", "megakaryocyte", "lymphoid"],
						  "LMPP": ["myeloid", "lymphoid"]}

	high_data = data.obs[(data.obs["ClonalGroup.Prob"]>threshold)].copy()
	lineages_to_be_removed = ["intermediate","hsc"]
	subdata = high_data[~high_data.lineage.isin(lineages_to_be_removed)]
	global_frequency = subdata["lineage"].value_counts(normalize=True)
	clade_frequency = subdata[["lineage", "ClonalGroup"]].groupby(["lineage", "ClonalGroup"]).size().reset_index()
	clade_frequency.columns= ["lineage", "clade", "frequency"]
	clade_frequency["adj"] = clade_frequency.apply(lambda row: row.frequency/global_frequency[row.lineage], axis=1)
	clade_frequency["probability"] = clade_frequency.groupby("clade")["adj"].transform(lambda x: x/x.sum())
	clade_frequency_pivot = clade_frequency.pivot(index="clade", columns = "lineage", values="probability").reset_index()
	probabilities_hsc = high_data[high_data.lineage == "hsc"][["ClonalGroup"]].reset_index(names="barcode").merge(clade_frequency_pivot, how="left", left_on="ClonalGroup", right_on = "clade")
	probabilities_hsc.drop("clade", axis=1, inplace=True)
	probabilities_terminal = high_data[~high_data.lineage.isin(lineages_to_be_removed)][["ClonalGroup", "lineage"]]
	probabilities_terminal = pd.get_dummies(probabilities_terminal, columns = ["lineage"], dtype=float, prefix='', prefix_sep='').reset_index(names="barcode")
	intermediate_cells = high_data[high_data.lineage=="intermediate"][["ClonalGroup", "STD.CellType"]].reset_index(names="barcode")
	clade_frequency_pivot = clade_frequency.pivot(index="clade", columns="lineage", values="adj").reset_index()
	probabilities_intermediate = intermediate_cells.apply(lambda x: intermediate_probability(clade_frequency_pivot, reachable_lineages, x), axis=1)
	probabilities_intermediate = pd.concat((intermediate_cells[["barcode", "ClonalGroup"]], probabilities_intermediate), axis=1)

	total_probs = pd.concat([probabilities_intermediate, probabilities_terminal, probabilities_hsc], axis=0).set_index("barcode")
	total_probs = total_probs.loc[high_data.index.tolist()]	
	total_probs.to_csv(branch_probabilities_path, sep=",", header=True, index=True)

	# SELECT PALANTIR CELLS
	saving_path = os.path.join(data_path, "selected_cells_palantir.json")
	data.obs["states"] = data.obs["STD.CellType"].map(lambda x: get_state_lineage_tracing(x))
	nearest_cells = search_cells(data, embedding_key = "X_umap", grouping_key = "states", n_select=1)
	cells = {"initial": {"hsc": nearest_cells["initial"][0]}, "terminal": {"erythroid": nearest_cells["erythroid"][0], "megakaryocyte": nearest_cells["megakaryocyte"][0], "monocyte": nearest_cells["monocyte"][0], "T": nearest_cells["T"][0], "B": nearest_cells["B"][0], "NK": nearest_cells["NK"][0], "dendritic": nearest_cells["dendritic"][0]}}
	with open(os.path.join(saving_path), "w") as f:
		json.dump(cells,f)
		f.close()
	
	# SELECT CELLRANK CELLS
	saving_path = os.path.join(data_path, "selected_cells.json")
	nearest_cells = search_cells(data, embedding_key = "X_umap", grouping_key = "states", n_select=30)
	cells = {"initial": {"hsc": nearest_cells["initial"]}, "terminal": {"erythroid": nearest_cells["erythroid"], "megakaryocyte": nearest_cells["megakaryocyte"], "monocyte": nearest_cells["monocyte"], "T": nearest_cells["T"], "B": nearest_cells["B"], "NK": nearest_cells["NK"], "dendritic": nearest_cells["dendritic"]}}
	with open(saving_path, "w") as f:
		json.dump(cells,f)
		f.close()
