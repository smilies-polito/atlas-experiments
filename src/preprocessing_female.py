import os
import json
# import scvi
import pandas as pd
import numpy as np
import scanpy as sc
import muon as mu 
#from scvi.external import SysVI
from src.utils import search_cells
from anndata import AnnData
from scipy.sparse import csr_matrix

def retrieve_coordinates(rna:AnnData)-> pd.DataFrame:
	dataframe = rna.var.copy()
	dataframe["Chromosome"] = dataframe["interval"].map(lambda x: x.split(":")[0])
	dataframe = dataframe[~(dataframe.Chromosome=="NA")]
	dataframe["Start"] = dataframe["interval"].map(lambda x: x.split(":")[1].split("-")[0])
	dataframe["End"] = dataframe["interval"].map(lambda x: x.split(":")[1].split("-")[1])
	dataframe.Start = dataframe.Start.astype(int)
	dataframe.End = dataframe.End.astype(int)
	return dataframe[["Chromosome", "Start", "End"]]


if __name__=="__main__":
	seed= 42 
	np.random.seed(seed)

#	scvi.settings.seed = seed
	working_dir = "/scvemo"
	data_path = os.path.join(working_dir, "data", "female_gonads")	
	annotations_path = os.path.join(data_path, "celltypist_annotations.csv")
	annotations = pd.read_csv(annotations_path, sep="\t", header=0, index_col=0)
#	ovary_types = ["Gi", "Oi", "early_supporting", "preGC_I", "preGC_IIa", "preGC_IIb", "granulosa", "oogonia_STRA8", "oogonia_meiotic", "CoelEpi_LHX9", "OSE", "PGC", "pre_oocyte", "early_sPAX8", "oocyte"]
	
	data = sc.read_10x_mtx(os.path.join(data_path, "filtered_feature_bc_matrix"), gex_only= False, var_names = "gene_symbols")
	data_h5 = sc.read_10x_h5(os.path.join(data_path, "filtered_feature_bc_matrix.h5"))
	data_h5.var_names_make_unique()
	rna = data[:, data.var.feature_types=="Gene Expression"] 
	atac = data[:, data.var.feature_types=="Peaks"]	

	rna.var["mt"] = rna.var_names.str.startswith("MT-")
	rna.obs["individual"] = rna.obs_names.map(lambda x: x.split("-")[1])
	rna.obs["barcode"] = rna.obs_names.map(lambda x: x.split("-")[0])
	atac.obs["individual"] = atac.obs_names.map(lambda x: x.split("-")[1])
	rna.obs.individual = rna.obs.individual.astype(int)	
	atac.obs.individual = atac.obs.individual.astype(int)	

	info = pd.read_csv(os.path.join(data_path, "samples.csv"), header=0)
	info["individual"] = info.index.astype(int) + 1

	rna.obs = rna.obs.merge(info, how="left", left_on = "individual", right_on="individual")
	rna.obs = rna.obs[["individual", "barcode", "library_id", "age"]]
	rna.obs_names = rna.obs["barcode"] + "-" + rna.obs["individual"].astype(str)
	
	sc.pp.calculate_qc_metrics(rna, qc_vars = ["mt"], inplace=True, log1p=False)
	sc.pl.violin(rna, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], show=False, multi_panel=True, save = "_rna_female_qc.png")
	sc.pl.scatter(rna, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save = "_rna_female_qc.png")
	mask = (rna.obs["total_counts"] > 100) & (rna.obs["total_counts"]<100000) & (rna.obs["pct_counts_mt"] < 5)
	rna = rna[mask, ~ rna.var.mt]
	rna.var = pd.merge(rna.var, data_h5.var[["genome", "interval"]], how="left", left_index = True, right_index=True)

	features = retrieve_coordinates(rna)

	mu.atac.tl.locate_file(atac, file=os.path.join(data_path, "atac_fragments.tsv.gz"), key = "fragments")
	mu.atac.tl.nucleosome_signal(atac)
	mu.atac.tl.tss_enrichment(atac, features) 
	sc.pl.violin(atac, ["tss_score", "nucleosome_signal"], show=False, save = "_atac_female_qc.png")
	low_tss, high_tss = np.percentile(atac.obs["tss_score"], 5), np.percentile(atac.obs["tss_score"], 95)
	low_nucleo, high_nucleo = np.percentile(atac.obs["nucleosome_signal"], 5), np.percentile(atac.obs["nucleosome_signal"], 95)
	mask = (atac.obs.nucleosome_signal > low_nucleo) & (atac.obs.nucleosome_signal<high_nucleo) & (atac.obs.tss_score > low_tss)
	atac  = atac[mask, :]

	intersection =set(atac.obs_names).intersection(set(rna.obs_names))
	rna = rna[rna.obs_names.isin(intersection), :]
	atac = atac[atac.obs_names.isin(intersection), :]

	activity = mu.atac.tl.count_fragments_features(data=atac, features=features, stranded = False)

	if not(isinstance(activity.X, csr_matrix)):
		activity.X = csr_matrix(activity.X)

#	rna.obs = rna.obs.merge(annotations, left_index=True, right_index=True)
#	activity.obs = activity.obs.merge(annotations, left_index=True, right_index=True)
#	rna = rna[rna.obs["majority_voting"].isin(ovary_types)]
#	activity = activity[activity.obs["majority_voting"].isin(ovary_types)]

	sc.pp.normalize_total(rna,target_sum=1e4)
	sc.pp.log1p(rna)
	sc.pp.highly_variable_genes(rna)
	sc.pp.pca(rna, random_state = seed)
	sc.pl.pca_variance_ratio(rna, show=False, save = "_female_rna.png")
		
	sc.pp.normalize_total(activity)
	sc.pp.pca(activity, random_state=seed)
	sc.pl.pca_variance_ratio(activity, show=False, save="_female_activity.png")

	n_pcs_rna = 15
	n_pcs_activity = 10
	knn_rna = 30	
	knn_activity = 30 
	wnn = 30

	# BBKNN
#	sc.external.pp.bbknn(rna, batch_key="individual", use_rep="X_pca", n_pcs = n_pcs_rna)
#	sc.external.pp.bbknn(activity, batch_key="individual", use_rep="X_pca", n_pcs = n_pcs_activity)
#	sc.tl.umap(rna, random_state = seed)
#	sc.tl.umap(activity, random_state = seed)
#	sc.pl.embedding(rna, basis="X_umap", color="individual", save="_rna_female_bbknn_individual.png")
#	sc.pl.embedding(activity, basis="X_umap", color="individual", save="_activity_female_bbknn_individual.png")

	#HARMONY
#	rna.obs["individual"] = rna.obs["individual"].astype("category")
#	activity.obs["individual"] = activity.obs["individual"].astype("category")
#	sc.external.pp.harmony_integrate(rna, key="individual", basis="X_pca", adjusted_basis="X_pca_harmony")
#	sc.external.pp.harmony_integrate(activity, key="individual", basis="X_pca", adjusted_basis="X_pca_harmony")
#	sc.pp.neighbors(rna,  n_neighbors=knn_rna, use_rep="X_pca_harmony", random_state = seed)
#	sc.pp.neighbors(activity,  n_neighbors=knn_activity, use_rep="X_pca_harmony", random_state = seed)
#	sc.tl.umap(rna, random_state = seed)
#	sc.tl.umap(activity, random_state = seed)
#	sc.pl.embedding(rna, basis="X_umap", color="individual", save="_rna_female_harmony_individual.png")
#	sc.pl.embedding(activity, basis="X_umap", color="individual", save="_activity_female_harmony_individual.png")

	# SCVI 
#	SysVI.setup_anndata(adata=activity, batch_key="individual", layer=None)
#	model = SysVI(adata=activity, embed_categorical_covariates=True)	
#	model.train()
#	embedding_activity = model.get_latent_representation(adata=activity)
#	embedding_activity = sc.AnnData(embedding_activity, obs=activity.obs)
#	embedding_activity.write(os.path.join(data_path, "embedding_activity.h5ad"))
#	embedding_activity = sc.read_h5ad(os.path.join(data_path, "embedding_activity.h5ad"))
#	sc.pp.neighbors(embedding_activity, use_rep="X", random_state=seed, n_neighbors=knn_activity)
#	sc.tl.umap(embedding_activity, random_state = seed)
#	sc.pl.embedding(embedding_activity, "X_umap", color=["individual"], save="scvi_female_activity_individual.png")

#	embedding_rna = sc.read_h5ad(os.path.join(data_path, "embedding_rna.h5ad"))
#	annotations = pd.read_csv(os.path.join(data_path, "celltypist_predictions.csv"), sep="\t", index_col=0, header=0)
	
#	embedding_rna.obs = embedding_rna.obs.merge(annotations, how="left", left_index=True, right_index=True)
#	embedding_activity.obs = embedding_activity.obs.merge(annotations, how="left", left_index=True, right_index=True)
#	rna.obs = rna.obs.merge(annotations, how="left", left_index=True, right_index=True)
#	activity.obs = activity.obs.merge(annotations, how="left", left_index=True, right_index=True)

#	sc.pp.neighbors(embedding_rna, use_rep="X", random_state=seed, n_neighbors=knn_rna)
#	sc.tl.umap(embedding_rna, random_state = seed)
#	sc.pl.umap(embedding_rna, color="majority_voting", save = "_scvi_rna_mv.png")
#	sc.pl.umap(embedding_activity, color="majority_voting", save = "_scvi_activity_mv.png")

	annotations = pd.read_csv(annotations_path, sep="\t", header=0, index_col=0)
	data = mu.MuData({"rna":rna, "activity":activity})
#	data = mu.MuData({"rna": embedding_rna, "activity":embedding_activity})
	data.obs = data.obs.merge(annotations, how="left", left_index=True, right_index=True)

	mu.pp.neighbors(data, key_added="wnn", n_neighbors=wnn, random_state=seed)
	mu.tl.umap(data, random_state=seed, neighbors_key="wnn")
	mu.pl.umap(data, color=["rna:individual", "majority_voting"], save="_mudata_female.png")
	data.write(os.path.join(data_path, "data.h5mu"))

	# SELECT PALANTIR CELLS
	saving_path = os.path.join(data_path, "selected_cells_palantir.json")
	nearest_cells = search_cells(data, embedding_key="X_umap", grouping_cells="majority_voting", n_select=1)
	cells = {"initial":{"PGC":nearest_cells["PGC"][0]}}
	
	with open(saving_path, "w") as f:
		json.dump(cells, f)
		f.close()
	

	
	
