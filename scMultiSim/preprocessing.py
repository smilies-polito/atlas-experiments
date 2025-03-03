import os
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.sparse import csr_matrix

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	data_path = ...
	activity = pd.read_csv(..., sep="\t", header=0, index_col=0)
	spliced = pd.read_csv(..., sep="\t", header=0, index_col=0)
	unspliced = pd.read_csv(..., sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(..., sep="\t", header=0, index_col=0)

	activity = AnnData(X=csr_matrix(activity.values), obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
	
	total_rna = csr_matrix(spliced.values.T + unspliced.values.T)
	rna = AnnData(X=csr_matrix(total_rna), obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	rna.layers["spliced"] = spliced.values.T
	rna.layers["unspliced"] = unspliced.values.T
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)

	data = mu.MuData({"rna":rna, "activity":activity})
	sc.pp.normalize_total(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.pca(data["rna"], random_state=seed, use_highly_variable=False)
	sc.pl.pca_variance_ratio(data["rna"])

	sc.pp.normalize_total(data["activity"])
	sc.pp.pca(data["activity"], random_state=seed, use_highly_variable=False)
	sc.pl.pca_variance_ratio(data["activity"])

	
	knn_rna = 30
	n_pcs_rna = 20
	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, n_pcs=n_pcs_rna, random_state=seed)
	
	n_pcs_activity = 10
	knn_activity = 30
	sc.pp.neighbors(data["activity"], n_neighbors=knn_activity, n_pcs = n_pcs_activity, random_state = seed)
	
	knn_multimodal = 30
	mu.pp.neighbors(data, n_neighbors=knn_multimodal, random_state=seed, key_added="wnn")
	mu.tl.umap(data, random_state=seed, neighbors_key="wnn")
	mu.pl.embedding(data, basis="X_umap", color=["rna:pop", "rna:pseudotime"])

	data.write(os.path.join(os.getcwd(), "data.h5mu"))
