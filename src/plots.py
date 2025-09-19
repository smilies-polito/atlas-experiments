import os
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from muon import MuData
from anndata import AnnData
from scipy.sparse import issparse
from src.utils import _check_keys
from matplotlib.patches import Patch 
from typing import Optional, Union, List

def plot_similarity_matrix(similarity_matrix, cell_types, path):
	sort_idx = np.argsort(cell_types)
	sorted_types = cell_types[sort_idx]
	sorted_matrix = similarity_matrix[sort_idx][:, sort_idx]

	unique_types = np.unique(sorted_types)
	type_palette = sns.color_palette("hls", len(unique_types))
	type_colors = dict(zip(map(str, unique_types), type_palette))
	row_colors = pd.Series(sorted_types.astype(str)).map(type_colors).to_numpy()
	
	g = sns.clustermap( sorted_matrix, 
			row_cluster=False, 
			col_cluster= False,	
			row_colors = row_colors,
			col_colors = row_colors,
			cmap = "viridis",
			figsize = (12,12),
			vmin= 0, vmax=1,
			cbar_pos=(0.91, 0.3, 0.02, 0.4),
		)
	handles = [Patch(facecolor=type_colors[t], label=t) for t in unique_types]
	g.ax_col_dendrogram.legend(
		handles=handles,
		title="Cell Types",
		loc="lower center",
		ncol=len(unique_types))

	plt.savefig(path)
	plt.close()



def plot_diffusion_space(diffusion_space, cell_types, path):
	sort_idx = np.argsort(cell_types)
	sorted_types = cell_types[sort_idx]
	sorted_matrix = diffusion_space[sort_idx, :]
	
	unique_types = np.unique(sorted_types)
	type_palette = sns.color_palette("hls", len(unique_types))
	type_colors = dict(zip(map(str, unique_types), type_palette))
	row_colors = pd.Series(sorted_types.astype(str)).map(type_colors).to_numpy()
	g = sns.clustermap( diffusion_space, 
			row_cluster=False, 
			col_cluster= False,	
			row_colors = row_colors,
			col_colors = None,
			cmap = "viridis",
			figsize = (12,12),
			vmin= 0, vmax=1
		)

	handles = [Patch(facecolor=type_colors[t], label=t) for t in unique_types]
	g.ax_col_dendrogram.legend(
		handles=handles,
		title="Cell Types",
		loc="center",
		ncol=len(unique_types))

	plt.savefig(path)
	plt.close()




def plot_probabilities(fates: pd.DataFrame, terminal_states: np.array, embedding: np.ndarray, terminal_locations: np.ndarray, save:bool = True, saving_path: str = None):
		
	x = embedding[:,0]
	y = embedding[:,1]
	if len(terminal_states)==0:
		print("WARNING - No terminal states detected!")
		path = os.path.join(saving_path, "no_probabilities.png") if saving_path is not None else saving_path 
		simple_scatter(x=x, y=y, save= save, saving_path = path, title = "No Terminal States", xticks=[], yticks=[])  
		return

	for idx, terminal in enumerate(terminal_states):
		plt.figure(figsize=(8,6))
		scatter = plt.scatter(x, y, c=fates[terminal], cmap="YlOrBr")
		plt.xticks([])
		plt.yticks([])
		plt.title(f"Fates towards {terminal}")
		plt.colorbar(scatter, label="probability")
		
		terminal_x, terminal_y = terminal_locations[idx]
		plt.scatter(terminal_x, terminal_y, color="red")

		if save and saving_path is not None:
			plt.savefig(os.path.join(saving_path, f"{terminal}_fates.png"))
		plt.close()


def plot_entropy(entropy: pd.Series, embedding: np.ndarray, save:bool = True, saving_path:str= None):
	x = embedding[:,0]
	y = embedding[:,1]
	path = os.path.join(saving_path, "entropy.png") if saving_path is not None else None
	
	simple_scatter(x=x, y=y, c=entropy.values, save=save, saving_path= path, title=f"Differentiation Potential", cbar_label= "potential", xticks=[], yticks=[])


def plot_pseudotime(pseudotime: pd.Series, embedding: np.ndarray, save:bool=True, saving_path:str=None):
	x = embedding[:,0]
	y = embedding[:,1]
	path = os.path.join(saving_path, "pseudotime.png") if saving_path is not None else None
	
	simple_scatter(x = x, y= y, c=pseudotime.values, save=save, saving_path= path, title= "Palantir Pseudotime", cbar_label= "pseudotime", xticks=[], yticks=[]) 


def plot_palantir_results(data: Union[AnnData, MuData], modality_key: str = None,
							embedding_key: str = "X_umap",
							pseudo_time_key: str = "palantir_pseudotime",
							entropy_key: str = "palantir_entropy",
							fate_prob_key: str = "palantir_fate_probabilities",
							save:bool = True,
							saving_path: Optional[str]=None):

	_check_keys(data, modality_key=modality_key, embedding_key = embedding_key, pseudo_time_key = pseudo_time_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key)
	embedding = data.obsm[embedding_key]
	data = data if modality_key is None else data[modality_key]

	fates = data.obsm[fate_prob_key].copy()
	terminal_states = fates.columns.values
	terminal_locations = embedding[np.where(fates.index.isin(terminal_states))[0]]
	plot_probabilities(fates=fates, terminal_states = terminal_states, terminal_locations = terminal_locations, embedding= embedding, save=save, saving_path = saving_path)
	
	plot_entropy(data.obs[entropy_key], embedding, save=save, saving_path = saving_path)
	plot_pseudotime(data.obs[pseudo_time_key], embedding, save=save, saving_path=saving_path)


def simple_scatter(x,y, c=None, categorical: bool= False, save:bool=True, saving_path:str=None, **kwargs):
	plt.figure(figsize=(8,6))

	if c is not None:
		if categorical:
			scatter = plt.scatter(x=x, y=y, c=c.cat.codes, cmap="tab20", s=10)
			labels = c.cat.categories
			legend_loc = kwargs.get("legend_loc", "best")
			legend_ncols = kwargs.get("legend_ncols", 1)
			handles, _ = scatter.legend_elements()
			plt.legend(handles, labels, loc=legend_loc, ncols = legend_ncols)
		else:
			scatter = plt.scatter(x=x, y=y, c=c, cmap="YlOrBr", s=10)
			clabel = kwargs.get("cbar_label", "")
			plt.colorbar(scatter, label=clabel)
	else:
		scatter = plt.scatter(x=x, y=y, s=10)

	if "title" in kwargs:
		plt.title(kwargs["title"])
	if "xlabel" in kwargs:
		plt.xlabel(kwargs["xlabel"])
	if "ylabel" in kwargs:
		plt.ylabel(kwargs["ylabel"])
	
	if "xticks" in kwargs:
		plt.xticks(kwargs["xticks"])
	if "yticks" in kwargs:
		plt.yticks(kwargs["yticks"])
	
	if "xlim" in kwargs:
		plt.xlim(kwargs["xlim"])
	if "ylim" in kwargs:
		plt.ylim(kwargs["ylim"])
		
	if save and saving_path is not None:
		try:
			plt.savefig(saving_path)
		except Exception as e:
			print("Figure not saved. check path", e)
	plt.close()


def plot_heatmap(data: Union[AnnData, MuData], similarity_key:str="connectivities", group_key:Union[str, List[str]]="celltype", save:Optional[str]=None, subset_key:Optional[str]=None, keep_subset:Optional[List[str]]=None):
	print(f"CALLED {similarity_key}")
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
			print(f"{missing} not in data.obs")
			group_key = list(set(group_key).difference(set(missing)))

	if subset_key is not None and subset_key not in group_key:
		raise KeyError(f"{subset_key} not in group_keys")

	if issparse(W):
		W= W.toarray()
	if isinstance(W, pd.DataFrame):
		W = W.to_numpy()

	print(f"W: {W.shape}")
	if subset_key is not None:
		mask = data.obs[subset_key].isin(keep_subset).values 
		W = W[mask, :]
	else:
		mask = np.ones(W.shape[0], dtype=bool)

	sorted_obs_rows = data.obs[mask].sort_values(group_key)
	sorted_obs_cols = data.obs.sort_values(group_key)
	sorted_idx_rows = data.obs[mask].index.get_indexer(sorted_obs_rows.index)
	sorted_idx_cols = data.obs.index.get_indexer(sorted_obs_cols.index)
	print(f"Checks, rows {sorted_idx_rows.shape} col {sorted_idx_cols.shape}")
	
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

