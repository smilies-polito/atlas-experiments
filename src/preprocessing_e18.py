import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import muon.atac as ac
from muon import MuData
from src.utils import retrieve_ensembl_coordinates

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	rng = np.random.default_rng(seed)

	working_dir = "/scvemo"
	data_path = os.path.join(working_dir, "data", "e18_mouse")
	saving_palantir_cells_path = os.path.join(data_path, "selected_cells_palantir.json")
	cc_genes_path = os.path.join(working_dir, "data", "genes_cellcycle.tsv")
	cc_genes = pd.read_csv(cc_genes_path, header=0, index_col=False, sep ="\t")
	print(cc_genes.head(3))
	
	annotations = pd.read_csv(os.path.join(data_path, "cell_annotations.tsv"), sep ="\t", header=0, index_col=0)
	annotations = annotations[~annotations.celltype.isin(['Interneurons2','Interneurons3',"Interneurons1", 'Cajal-Retzius','Microglia'])]
	print(annotations.head(2))
	data = sc.read_10x_mtx(os.path.join(data_path,  "filtered_feature_bc_matrix"), var_names="gene_symbols", gex_only=False)
	atac = data[:, data.var["feature_types"]=="Peaks"]
	rna = data[:, data.var["feature_types"]=="Gene Expression"]
	print(f"rna: {rna.shape}, atac:{atac.shape}")
	rna.var_names_make_unique()

	# RNA QC FILTERING 
	sc.pp.calculate_qc_metrics(rna, inplace=True)
	sc.pl.violin(rna, ["n_genes_by_counts", "total_counts"], show=False, save = "_e18_rna_qc.png")	
	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=10000)
	
	# RETRIEVE COORDINATES FOR GENES USING ENSEMBL
	skip_chromosomes = ['JH584299.1','GL456221.1','GL456219.1', "chrMT"]
	coordinates = retrieve_ensembl_coordinates(list(rna.var_names), skip_chromosomes = skip_chromosomes)
	rna = rna[:, rna.var_names.isin(coordinates["Symbol"])]
	print(f"New rna shape is {rna.shape}")

	# ATAC PREPROCESSING
	ac.tl.locate_file(atac, file=os.path.join(data_path, "e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz"), key="fragments")
	ac.tl.nucleosome_signal(atac)
	ac.tl.tss_enrichment(atac, coordinates)
	sc.pl.violin(atac, ["tss_score", "nucleosome_signal"], show=False, save = "_e18_atac_qc.png")
	low_tss, high_tss = np.percentile(atac.obs["tss_score"], 5), np.percentile(atac.obs["tss_score"], 95)
	low_nucleo, high_nucleo = np.percentile(atac.obs["nucleosome_signal"], 5), np.percentile(atac.obs["nucleosome_signal"], 95)
	conditions = (atac.obs.nucleosome_signal > low_nucleo) & (atac.obs.nucleosome_signal < high_nucleo) & (atac.obs.tss_score > low_tss)
	atac = atac[conditions, :]
	print(f"New atac.shape is {atac.shape}")

	intersection = set(rna.obs_names).intersection(set(atac.obs_names)).intersection(set(list(annotations.index.values)))
	atac = atac[atac.obs_names.isin(intersection), :]
	rna = rna[rna.obs_names.isin(intersection), :]
	print(f"After intersection rna: {rna.shape}, atac: {atac.shape}")

	sc.pp.normalize_total(rna, target_sum=1e4)
	sc.pp.log1p(rna)
	sc.pp.scale(rna)
	sc.pp.highly_variable_genes(rna)
	rna.var_names = rna.var_names.str.upper()

	s_genes = list(set(cc_genes["G1/S"].values).intersection(set(rna.var_names)))
	g2m_genes = list(set(cc_genes["G2/M"].values).intersection(set(rna.var_names)))
	sc.tl.score_genes_cell_cycle(rna, s_genes=s_genes, g2m_genes=g2m_genes)
	sc.pp.regress_out(rna, ["S_score", "G2M_score"])

	# ACITIVITY
	activity = mu.atac.tl.count_fragments_features(data = atac, features = coordinates, stranded=True)
	sc.pp.normalize_total(activity)
	activity.var_names = activity.var_names.str.upper()

	data = MuData({"rna":rna, "activity":activity})
	data.obs = data.obs.merge(annotations, left_index=True, right_index=True, how="left") 
	print(f"Data: {data.shape}")

	sc.pp.pca(data["rna"], random_state=seed)
	sc.pp.pca(data["activity"], random_state=seed)
	sc.pl.pca_variance_ratio(data["rna"], show=False, save="rna_e18.png")
	sc.pl.pca_variance_ratio(data["activity"], show=False, save="activity_e18.png")
	
	pcs_rna = 20
	pcs_activity = 10
	knn_rna = 20
	knn_activity = 20
	wnn = 20

	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, n_pcs= pcs_rna, random_state=seed)
	sc.pp.neighbors(data["activity"], n_neighbors=knn_activity, n_pcs=pcs_activity, random_state = seed)
	mu.pp.neighbors(data, key_added = "wnn", n_neighbors = wnn, random_state = seed)
	mu.tl.umap(data, random_state = seed, neighbors_key = "wnn")
	mu.pl.umap(data, color = ["celltype"], save= "_e18_multiomics.png")
	
	data.write(os.path.join(data_path, "data.h5mu"))

	# Select palantir states
	cells = {}
	cells["initial"] = {"rg": np.random.choice(data.obs[data.obs["celltype"]=="RG, Astro, OPC"].index)}
	with open(saving_palantir_cells_path, "w") as f:
		json.dump(cells,f)
		
	

	
