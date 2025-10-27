import os, gc, re
import pandas as pd
import numpy as np
import muon as mu
import argparse
import json
import scanpy as sc
from src.utils import compute_skeletal_activity, search_cells

def deduplicate_features(features:pd.DataFrame) -> pd.DataFrame:
	features = features[features["Feature"]=="gene"].copy()
	features["Start"] = features["Start"].astype("Int64")
	features["End"] = features["End"].astype("Int64")
	features["length"] = (features["End"] - features["Start"]).astype("Int64")
	features["level"] = features["attribute"].str.extract(r'level\s+"([^"]+)"', expand=False).astype("Float64")
	features["level_rank"] = pd.to_numeric(features["level"]).fillna(9).astype("Int64")
	features["biotype_rank"] = (features["gene_type"] != "protein_coding").astype("Int64")
	f = features.sort_values(["gene_name", "biotype_rank", "level_rank", "length"], ascending=[True, True, True, False])
	cols_to_keep = ["Chromosome", "Start", "End", "Strand", "gene_name"]
	return f.drop_duplicates(subset=["gene_name"], keep="first")[cols_to_keep]


def map_to_lineage(celltype:str) -> str:
	# knee hip shoulder
	dictionary = {
		"mesenchymal": ["HIC1+Mes","LimbMes","LEPR+Mes"],
		"osteoblast": ["Preosteoblast", "Osteoblast", "Osteocyte", "MatureOsteocyte"], 
		"fibroblast": ["PERI", "TENO", "SynFIB", "PAX7+ Myo", "MYH3+ Myo", "MyofibPRO", "PerineuralFIB", "FibroPRO1", "FibroPRO2", "DermFIB1",     "DermFIB2"],
		"chondrocyte" : ["ArticularChon1", "ArticularChon2", "ChondroPro1", "ChondroPro2", "CyclingChon", "DLK1hiChon", "HypertrophicChon", "MaturingChon", "PAX7hiChon", "InterzoneChon"] 
	}
	for k,v in dictionary.items():
		if celltype in v:
			return k 
	return None


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	working_dir = "/scvemo"
	data_dir = os.path.join(working_dir, "data", "skeletalDev")
	output = os.path.join(working_dir, "output", "skeletalDev")
	
#	data = sc.read_h5ad(os.path.join(data_dir, "Whole_ATAC_326532_564709.h5ad"))
#	rna = data[:, data.var["modality"]=="Gene Expression"]
#	atac = data[:, data.var["modality"]=="Peaks"]

	parser = argparse.ArgumentParser()
	parser.add_argument("--site", type=str)
	parser.add_argument("--lineage", type=str)
	args = parser.parse_args()
	site = args.site 
	lineage = args.lineage
	print(f"Site = {site}, Lineage = {lineage}")

#	rna = sc.read_h5ad(os.path.join(data_dir, f"rna_{site}.h5ad"))
#	atac = sc.read_h5ad(os.path.join(data_dir, f"atac_{site}.h5ad"))
#	rna = rna[rna.obs.anatomical_site == site].copy()
#	atac = atac[atac.obs.anatimocal_site == site].copy()

	# Rimuovo cellule non di sviluppo osseo/cartilagine in hip, knee, shoulder
#	mesenchymal = ["HIC1+Mes","LimbMes","LEPR+Mes"]
#	osteoblast = ["Preosteoblast", "Osteoblast", "Osteocyte", "MatureOsteocyte"]
#	fibroblasts = ["PERI", "TENO", "SynFIB", "PAX7+ Myo", "MYH3+ Myo", "MyofibPRO", "PerineuralFIB", "FibroPRO1", "FibroPRO2", "DermFIB1", "DermFIB2"] 
#	chondrocytes = ["ArticularChon1", "ArticularChon2", "ChondroPro1", "ChondroPro2", "CyclingChon", "DLK1hiChon", "HypertrophicChon", "MaturingChon", "PAX7hiChon", "InterzoneChon"] 
	
#	whole_system = mesenchymal + osteoblast + fibroblasts + chondrocytes
#	rna = rna[rna.obs.Celltype_fig1.isin(whole_system)].copy()
#	atac = atac[atac.obs.Celltype_fig1.isin(whole_system)].copy()

#	rna.obs["lineage"] = rna.obs.Celltype_fig1.map(lambda x: map_to_lineage(x)) 

	# Aggiungo info di Chromosome + Start + End per ogni gene 
#	rna.var["gene_name"] = pd.Index(rna.var_names).str.replace(r"\.\d+$", "", regex=True)	
#	gtf = pd.read_csv(os.path.join(data_dir, "refdata-cellranger-arc-GRCh38-2020-A-2.0.0", "genes", "genes.gtf"), sep="\t", header=None, index_col=None, comment="#")
#	columns = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "attribute"]
#	gtf.columns = columns 
#	gtf["gene_name"] = gtf["attribute"].str.extract(r'gene_name\s+"([^"]+)"', expand=False)
#	gtf["gene_type"] = gtf["attribute"].str.extract(r'gene_type\s+"([^"]+)"', expand=False)
#	gtf = deduplicate_features(gtf)
#	var_names = rna.var.index
#	rna.var = rna.var.merge(gtf, how="left", left_on="gene_name", right_on="gene_name", validate="many_to_one")
#	rna.var.index = var_names
#	mask = (rna.var.Chromosome.isna()) & (rna.var.Chromosome=="chrM")
#	rna = rna[:, ~mask].copy()

	# Ricavo peaks Chromosome Start End per calcolare attività
#	tmp = atac.var_names.to_series().str.split(':|-', expand=True)
#	tmp.columns = ["Chromosome", "Start", "End"]
#	tmp.astype({"Start":"Int64", "End":"Int64"})
#	atac.var = atac.var.join(tmp, how="left")
#	features = rna.var[["Chromosome", "Start", "End", "Strand"]]
#	activity = compute_skeletal_activity(features=features, atac=atac, stranded=True)
#	activity.obs = activity.obs.merge(rna.obs[["run_id"]], how="left", left_index=True, right_index=True)

	# Whole system
#	data = mu.MuData({"rna": rna, "activity":activity})

#	if lineage == "whole":
#		path = os.path.join(data_dir, f"data_{site}.h5mu")
#		data.write_h5mu(path)
#	else:
#		path = os.path.join(data_dir, f"data_{site}_{lineage}.h5mu")
#		lineages = [lineage, "mesenchymal"]
#		data =  data[data.obs["rna:lineage"].isin(lineages)].copy()	
#		data.write_h5mu(path)
	

	# RNA preprocessing 
#	sc.pp.normalize_total(data["rna"], target_sum=1e4)
#	sc.pp.log1p(data["rna"])
#	sc.pp.highly_variable_genes(data["rna"])
#	sc.pp.pca(data["rna"], random_state = seed)
#	sc.pl.pca_variance_ratio(data["rna"], show=False, save=f"rna_{site}_{lineage}.png")

#	sc.pp.normalize_total(data["activity"])
#	sc.pp.pca(data["activity"], random_state = seed)
#	sc.pl.pca_variance_ratio(data["activity"], show=False, save=f"activity_{site}_{lineage}.png")

#	n_pcs_rna = 15
#	n_pcs_activity = 10
#	knn_rna = 30
#	knn_activity = 30 
#	wnn = 30

	# BBKNN
#	sc.external.pp.bbknn(data["rna"], batch_key="run_id", use_rep="X_pca", n_pcs = n_pcs_rna)
#	sc.external.pp.bbknn(data["activity"], batch_key="run_id", use_rep="X_pca", n_pcs = n_pcs_activity)
#	sc.tl.umap(data["rna"], random_state = seed)
#	sc.tl.umap(data["activity"], random_state = seed)
#	sc.pl.embedding(data["rna"], basis="X_umap", color="run_id", save=f"_rna_bbknn_{site}_{lineage}.png")
#	sc.pl.embedding(data["activity"], basis="X_umap", color="run_id", save=f"_activity_bbknn_{site}_{lineage}.png")

	#HARMONY
#	data["rna"].obs["run_id"] = data["rna"].obs["run_id"].astype("category")
#	data["activity"].obs["run_id"] = data["activity"].obs["run_id"].astype("category")
#	sc.external.pp.harmony_integrate(data["rna"], key="run_id", basis="X_pca", adjusted_basis="X_pca_harmony")
#	sc.external.pp.harmony_integrate(data["activity"], key="run_id", basis="X_pca", adjusted_basis="X_pca_harmony")
#	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, use_rep="X_pca_harmony", random_state=seed)
#	sc.pp.neighbors(data["activity"], n_neighbors=knn_activity, use_rep="X_pca_harmony", random_state=seed)
#	sc.tl.umap(data["rna"], random_state = seed)
#	sc.tl.umap(data["activity"], random_state = seed)
#	sc.pl.embedding(data["rna"], basis="X_umap", color="run_id", save=f"_rna_harmony_{site}_{lineage}.png")
#	sc.pl.embedding(data["activity"], basis="X_umap", color="run_id", save=f"_activity_harmony_{site}_{lineage}.png")
	

#	mu.pp.neighbors(data, key_added="wnn", n_neighbors=wnn, random_state=seed)
#	mu.tl.umap(data, random_state = seed, neighbors_key="wnn")
#	mu.pl.umap(data, color=["rna:lineage", "rna:Celltype_fig1"], save = f"_harmony_{site}_{lineage}.png")
	
	path = os.path.join(data_dir, f"bbknn_{site}.h5mu") if lineage == "whole" else os.path.join(data_dir, f"bbknn_{site}_{lineage}.h5mu")
	data = mu.read_h5mu(path)
	data.obs["composite"] = data["rna"].obs["lineage"].astype(str) + ":" + data["rna"].obs["pcw"].astype(str)
#	data.write_h5mu(path) 

	# SELECT PALANTIR INITIAL CELL
	saving_appendix = f"selected_cells_palantir_bbknn_{site}.json" if lineage == "whole" else f"selected_cells_palantir_bbknn_{site}_{lineage}.json"
	saving_path = os.path.join(data_dir, saving_appendix)
	nearest_cells = search_cells(data, embedding_key="X_umap", grouping_key="composite", n_select=1)
	cells = {"initial":{"initial":nearest_cells["mesenchymal:5.7"][0]}}
	with open(saving_path, "w") as f:
		json.dump(cells,f)
		f.close()

	



	
	
	
