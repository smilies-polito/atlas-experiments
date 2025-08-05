import numpy as np
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from muon import MuData
from matplotlib.patches import Patch
from scipy.sparse import issparse
from typing import Optional, Union, List

def plot_group_heatmap(data: Union[MuData, AnnData], similarity_key:str="connectivities", group_key:Union[list, List[str]]="celltype", save:Optional[str]=None, subset_key:Optional[str]=None, keep_subset:Optional[List[str]]=None):

	if subset_key is not None and subset_key not in data.obs.columns:
		raise KeyError(f"{subset_key} not in data.obs")
	if subset_key is not None and keep_subset is not None:
		keep_subset = [ks for ks in keep_subset if ks in data.obs[subset_key].unique()]
		if len(keep_subset) == 0:
			raise ValueError(f"key_subset not valid")

	if not similarity_key in data.obsp:
		if not similarity_key in data.obsm:
			raise KeyError(f"{similarity_key} not in data.obsm or data.obsp")
		W = data.obsm[similarity_key]
		manipulate_columns = False
	else:
		W = data.obsp[similarity_key]
		manipulate_columns = True

	if isinstance(group_key, str):
		if not group_key in data.obs.columns:
			raise KeyError(f"{group_key} not in data.obs")
		group_key = [group_key]
	else:
		missing = [k for k in group_key if k not in data.obs.columns]
		if len(missing)>0:
			raise KeyError(f"{missing} not in data.obs")
			group_key = list(set(group_key).difference(set(missing)))

	if subset_key is not None and subset_key not in group_key:
		raise KeyError(f"{subset_key} not in group_keys")
	
	if issparse(W):
		W= W.toarray()
	if isinstance(W, pd.DataFrame):
		W = W.to_numpy()

	if subset_key is not None:
		mask = data.obs[subset_key].isin(keep_subset).values 
		W = W[:, mask]
	else:
		mask = np.ones(W.shape[0], dtype=bool)

	sorted_obs_rows = data.obs.sort_values(group_key)
	sorted_obs_cols = data.obs[mask].sort_values(group_key)
	sorted_idx_rows = data.obs.index.get_indexer(sorted_obs_rows.index)
	sorted_idx_cols = data.obs[mask].index.get_indexer(sorted_obs_cols.index)
	
	if manipulate_columns:
		W_sorted = W[sorted_idx_rows, :][:, sorted_idx_cols]
	else:
		W_sorted = W[sorted_idx_rows, :]
	
	luts = {}
	row_colors = []
	col_colors = []

	for key in group_key:
		cat = data.obs[key].astype("category")	
		palette = sns.color_palette("tab20", len(cat.cat.categories))
		lut = dict(zip(cat.cat.categories, palette))
		luts[key] = lut
		color_row = [lut[val] for val in data.obs.iloc[sorted_idx_rows][key]]
		row_colors.append(color_row)

	if manipulate_columns and W.shape[0]==W.shape[1]:
		col_colors = row_colors 
	elif manipulate_columns and W.shape[1] < W.shape[0]:
		col_luts = luts.copy() 
		col_luts[subset_key] = {key: values for key,values in col_luts[subset_key].items() if key in keep_subset}
		for key in group_key:	
			color_col = [col_luts[key][val] for val in data.obs[mask].iloc[sorted_idx_cols][key]]
			col_colors.append(color_col)
	else:
		col_colors = None

	sns.set(style="white")
	g = sns.clustermap(
  	 W_sorted,
	 row_cluster=False,
   	 col_cluster=False,
   	 row_colors=row_colors,
  	 col_colors=col_colors,
	 cmap="viridis",
  	 xticklabels=False,
	 yticklabels=False,
	 figsize=(20,20)
	)
	
	for key in group_key:
		for label in luts[key]:
			g.ax_col_dendrogram.bar(0,0, color = luts[key][label], label = f"{key}:{label}", linewidth =0)

	g.ax_col_dendrogram.legend(loc="center", ncol = 4)


	if save is not None:
		plt.savefig(save)
		plt.close()



def scatter_3D(data:Union[MuData, AnnData], key:str, color_key:Optional[str]=None):
	if key not in data.obsm:
		if key not in data.obsp:
			raise KeyError(f"{key} not in data.obsp nor data.obsm")
		X = data.obsp[key]
	else:
		X = data.obsm[key]

	if color_key is not None:
		if color_key not in data.obs.columns:
			raise KeyError(f"{color_key} not in data.obs")
		cat = data.obs[color_key].astype("category")	
		palette = sns.color_palette("tab20", len(cat.cat.categories))
		lut = dict(zip(cat.cat.categories, palette))
		color = [lut[val] for val in data.obs[color_key]]
		legend_elements = [Patch(facecolor=lut[cat], label=str(cat)) for cat in lut]
	else: 
		color = None
		legend_elements = None

	if issparse(X):
		X = X.toarray()
	if isinstance(X, pd.DataFrame):
		X = X.to_numpy()

	if X.shape[1] < 3:
		raise ValueError(f"X.shape[1] expected to be higher than 3")

	fig= plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(X[:,0], X[:,1], X[:,2], c=color)	

	if legend_elements:
		ax.legend(handles=legend_elements, title=color_key, loc="upper right")

	ax.set_xlabel(f"{key} [1]")
	ax.set_ylabel(f"{key} [2]") 
	ax.set_zlabel(f"{key} [3]") 
	plt.show()


def plot_umap(data: Union[AnnData, MuData], key:str, color_key:Optional[str]=None, save:Optional[str]=None, seed:int = 42):
	if key not in data.obsm:
		if key not in data.obsp:
			raise KeyError(f"{key} not in data.obsp nor data.obsm")
		X = data.obsp[key]
	else:
		X = data.obsm[key]
	
	if issparse(X):
		X = X.toarray()
	if isinstance(X, pd.DataFrame):
		X = X.to_numpy()

	reducer = umap.UMAP(n_components=2, random_state = seed)
	X_umap = reducer.fit_transform(X)

	if color_key is not None:
		if color_key not in data.obs.columns:
			raise KeyError(f"{color_key} not in data.obs")
		cat = data.obs[color_key].astype("category")	
		palette = sns.color_palette("tab20", len(cat.cat.categories))
		lut = dict(zip(cat.cat.categories, palette))
		color = [lut[val] for val in data.obs[color_key]]
		legend_elements = [Patch(facecolor=lut[cat], label=str(cat)) for cat in lut]
	else: 
		color = None
		
	fig= plt.figure(figsize=(12,12))
	plt.scatter(X_umap[:,0], X_umap[:,1], c=color, s=10)	

	if legend_elements:
		plt.legend(handles=legend_elements, title=color_key, loc="upper right")

	plt.xlabel(f"UMAP 1")
	plt.ylabel(f"UMAP 2") 

	if save:
		plt.savefig(save)

