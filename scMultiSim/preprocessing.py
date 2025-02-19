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
	data_path = 
	atac = pd.read_csv( ... ,sep="\t", header=0, index_col=0)
	spliced = pd.read_csv( ... , sep="\t", header=0, index_col=0)
	unspliced = pd.read_csv( ... , sep="\t", header=0, index_col=0)
	metadata = pd.read_csv( ... , sep="\t", header=0, index_col=0)

	atac = AnnData(X=csr_matrix(atac.values.T), obs = pd.DataFrame(data=None, index=atac.columns, columns=None), var = pd.DataFrame(data=None, index=atac.index.values, columns=None))
	atac.obs = atac.obs.merge(metadata, how="left", left_index=True, right_index=True)
	
	total_rna = csr_matrix(spliced.values.T + unspliced.values.T)
	rna = AnnData(X=csr_matrix(total_rna), obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	rna.layers["spliced"] = spliced.values.T
	rna.layers["unspliced"] = unspliced.values.T
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)

	data = mu.MuData({"rna":rna, "atac":atac})
	sc.pp.normalize_total(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.pca(data["rna"], random_state=seed, use_highly_variable=False)
	
	knn_rna = 10
	n_pcs = 10
	sc.pp.neighbors(data["rna"], n_neighbors=knn_rna, n_pcs=n_pcs, random_state=seed)

	mu.atac.pp.tfidf(data["atac"])
	mu.atac.tl.lsi(data["atac"])
	
	n_lsi = 10
	knn_atac = 10
	# remove first component as usual (high correlation with sequencing depth) 
	data["atac"].obsm["X_lsi"] = data["atac"].obsm["X_lsi"][:, 1:]
	data["atac"].uns["lsi"]["stdev"] = data["atac"].uns["lsi"]["stdev"][1:]
	data["atac"].varm["LSI"] = data["atac"].varm["LSI"][:, 1:

	sc.pp.neighbors(data["atac"], n_neighbors=knn_atac, n_pcs=n_lsi, use_rep="X_lsi")

	mu.pp.neighbors(data, n_neighbors=10, random_state=seed, key_added="wnn")
	mu.tl.umap(data, random_state=seed, neighbors_key="wnn")
	mu.pl.embedding(data, basis="X_umap", color=["rna:pop", "rna:pseudotime"])

	data.write(os.path.join(os.getcwd(), "data.h5mu"))
