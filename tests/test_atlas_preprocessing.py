import os
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from atlas import ATLAS
from muon import MuData
from anndata import AnnData

# TRUE DATA CAN BE DOWNLOADED https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-2-0

if __name__ == "__main__":
	seed = 42
	np.random.seed(42)
	working_directory = os.getcwd()

	data_path = os.path.join(working_directory, "data", "embryonic_mouse_brain", "filtered_feature_bc_matrix")
	features_path = os.path.join(data_path, "features.tsv.gz")
	fragment_path = os.path.join(working_directory, "data", "embryonic_mouse_brain", "e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz")
	annotation_path = os.path.join(working_directory, "data", "embryonic_mouse_brain", "cell_annotations.tsv")
	features = pd.read_csv(features_path, sep="\t", index_col = 0, header = None)
	features.columns = ["name", "type", "Chromosome", "start", "end"]
	features = features[features["type"] == "Gene Expression"]

	annotations = pd.read_csv(annotation_path, sep="\t", header=0, index_col=0)
	data = sc.read_10x_mtx(data_path, var_names = "gene_symbols", gex_only=False)
	atac = data[:, data.var["feature_types"] == "Peaks"]
	rna = data[:, data.var["feature_types"] == "Gene Expression"]
	rna.var_names_make_unique()
	rna.var = rna.var.reset_index(names="symbol").set_index("gene_ids").join(features, how="left").set_index("symbol")

	sc.pp.normalize_total(rna, target_sum=1e4)
	sc.pp.log1p(rna)
	sc.pp.highly_variable_genes(rna)
	sc.pp.pca(rna, random_state=seed)

	mudata = MuData({"rna": rna, "ATAC": atac})


	# Preprocessing specified with method = "palantir" but works in the same way for any other method
	atlas = ATLAS(mudata=mudata, fragment_path = fragment_path, random_state = seed, method="palantir")
	assert atlas._impl.atac_key == "atac"
	assert atlas._impl.rna_key == "rna"
	assert atlas._impl.activity_key is None
	assert not atlas._impl.use_activity 
	assert "files" in atlas._impl.mudata["atac"].uns
	assert "fragments" in atlas._impl.mudata["atac"].uns["files"] 
	assert atlas._impl.random_state == seed

	features = rna.var[["Chromosome", "start", "end"]]
	features["start"] = features["start"].astype(int)
	features["end"] = features["end"].astype(int)
	mask = (features["Chromosome"].notna() & \
			features["Chromosome"].str.startswith("chr") & \
			 ~ features["Chromosome"].isin(["chrX", "chrY"]))
	features = features[mask]
	
	atlas.preprocessing(n_pcs_rna = 20,
						n_pcs_act = 10,
						knn_rna = 30,
						knn_act = 30,
						n_neighbors = 30, 
						stranded = False, 
						features = features)

	assert atlas._impl.activity_key == "activity"
	assert "activity" in atlas._impl.mudata.mod
	assert "X_pca" in atlas._impl.mudata["activity"].obsm
	assert "distances" in atlas._impl.mudata["rna"].obsp
	assert "connectivities" in atlas._impl.mudata["rna"].obsp
	assert "distances" in atlas._impl.mudata["activity"].obsp
	assert "connectivities" in atlas._impl.mudata["activity"].obsp
	assert "wnn_distances" in atlas._impl.mudata.obsp
	assert "wnn_connectivities" in atlas._impl.mudata.obsp
	assert "X_umap" in atlas._impl.mudata.obsm
						

