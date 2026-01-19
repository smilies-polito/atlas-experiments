import os
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import scvelo as scv
import matplotlib.pyplot as plt
from itertools import product
from atlas import ATLAS
from anndata import AnnData
from scipy.sparse import csr_matrix

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	
	differentiation_tree = "phyla3" # Either phyla3 or phyla5 to fetch data from and save results anndata
	working_directory = os.getcwd() # set path to repository 
	data_path = os.path.join(working_directory, "data", f"{differentiation_tree}") 
	grid = {"diff_cif_fraction" : [.1, .3, .5, .7, .9], "cif_sigma": [.1, .3, .5, .7, .9]}
	knn_grid = {"knn_rna": [30, 50, 70, 100], "knn_activity": [30, 50, 70, 100], "wnn": [30, 50, 70, 100]}
	n_pcs_rna = 20
	n_pcs_activity = 10
	
	for diff_cif_fraction, cif_sigma in product(*grid.values()):
		# Lettura dei dati 
		activity = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
		spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
		unspliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_unspliced.tsv"), sep="\t", header=0, index_col=0)
		metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

		# creazione matrice di attività
		activity = AnnData(X=csr_matrix(activity.values), 
				obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
				var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
		# creazione matrice di rna	
		total_rna = csr_matrix(spliced.values.T)
		rna = AnnData(X=csr_matrix(total_rna), 
				obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
				var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
		rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
		
		# creazione oggetto MuData 
		data = mu.MuData({"rna":rna, "activity":activity})
		cls = ATLAS(mudata = data, method="palantir")
	
		# rna preprocessing
		sc.pp.normalize_total(data["rna"])
		sc.pp.log1p(data["rna"])
		sc.pp.pca(data["rna"], random_state=seed, use_highly_variable=False)

		# activity preprocessing
		sc.pp.normalize_total(data["activity"])
		sc.pp.pca(data["activity"], random_state=seed, use_highly_variable=False)
			
		for knn_rna, knn_activity, wnn in product(*knn_grid.values()):
			saving_folder = os.path.join(data_path, f"{knn_rna}_{knn_activity}_{wnn}") 
			if not os.path.exists(saving_folder):
				os.mkdir(saving_folder)
			sc.pp.neighbors(data["rna"], 
					n_neighbors=knn_rna, 
					n_pcs=n_pcs_rna, 
					random_state=seed)
			sc.pp.neighbors(data["activity"],
					n_neighbors=knn_activity, 
					n_pcs = n_pcs_activity, 
					random_state = seed)
			cls.preprocessing(n_neighbors=wnn)
			mu.tl.umap(data, random_state=seed, neighbors_key="wnn")
			mu.pl.embedding(data, 
					basis="X_umap", 
					color=["rna:pop", "rna:pseudotime"], 
					show=False, 
					save = f"{diff_cif_fraction}_{cif_sigma}_{knn_rna}:{knn_activity}:{wnn}.png" )
			
			data.write(os.path.join(saving_folder, f"{diff_cif_fraction}_{cif_sigma}.h5mu"))
