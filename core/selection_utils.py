import os
import numpy as np
import pandas as pd
import muon as mu 
from scipy.spatial.distance import cdist



def compute_centroids(data: mu.MuData, embedding_key: str="X_umap", grouping_key:str="celltype") -> pd.DataFrame:
	if not embedding_key in data.obsm:
		raise KeyError(f"{embedding_key} not in data.obsm")
	if not grouping_key in data.obs:
		raise KeyError(f"{grouping_key} not in data.obs")

	embedding = data.obsm[embedding_key]
	columns = [f"col{n}" for n in range(embedding.shape[1])]

	df = pd.DataFrame(embedding, index=data.obs_names, columns=columns)
	df[grouping_key] = data.obs[grouping_key]
	
	centroids = df.groupby(by=grouping_key).mean()
	return centroids


def find_nearest_cells(data: mu.MuData, centroids:pd.DataFrame, embedding_key: str="X_umap", grouping_key:str="celltype", n_select:int=30) -> dict:
	if not embedding_key in data.obsm:
		raise KeyError(f"{embedding_key} not in data.obsm")
	if not grouping_key in data.obs:
		raise KeyError(f"{grouping_key} not in data.obs")

	embedding = data.obsm[embedding_key]	

	nearest_cells = {}
	
	for group in centroids.index:
		mask = data.obs[grouping_key] == group
		group_coordinates = embedding[mask.values, :]
		group_observations = data.obs_names[mask.values]
		group_centroid = centroids.loc[group].values.reshape(1, -1)
		dists = cdist(group_coordinates, group_centroid).flatten()
		top_indices = np.argsort(dists)[:n_select]
		nearest_cells[group] = group_observations[top_indices].tolist()

	return nearest_cells 
	

def search_cells(data: mu.MuData, embedding_key:str="X_umap", grouping_key:str="celltype", n_select:int=1) -> dict:
	centroids = compute_centroids(data, embedding_key=embedding_key, grouping_key=grouping_key)
	nearest_cells = find_nearest_cells(data, centroids, embedding_key = embedding_key, grouping_key = grouping_key, n_select=n_select)
	return nearest_cells

