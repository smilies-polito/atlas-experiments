import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
from muon import MuData
from anndata import AnnData
from core.utils import simple_scatter
from .palantir_utils import _check_keys
from scipy.stats import pearsonr
from typing import Union, Optional

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


def plot_probabilities(fates: pd.DataFrame, terminal_states: np.array, embedding: np.ndarray, terminal_locations: np.ndarray, save:bool = True, saving_path: str = None):
		
	x = embedding[:,0]
	y = embedding[:,1]
	if len(terminal_states)==0:
		warnings.warn("No terminal states detected!")
		path = os.path.join(saving_path, "no_probabilities.png") if saving_path is not None else saving_path 
		simple_scatter(x=x, y=y, save= save, saving_path = path, title = "No Terminal States", xticks=[], yticks=[])  
		return

	for idx, terminal in enumerate(terminal_states):
		plt.figure(figsize=(8,6))
		scatter = plt.scatter(x, y, c=fates[terminal])
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
		


def pseudotime_correlation(data: MuData, modality_key: Optional[str]=None, group_key: str = "celltype", true_key: str = "pseudotime", infer_key: str = "palantir_pseudotime", save:bool=True, saving_path: str= None, epsilon:float = 0.01):	

	_check_keys(data=data, modality_key = modality_key, pseudo_time_key = true_key)
	_check_keys(data=data, modality_key = modality_key, pseudo_time_key = infer_key)
	_check_keys(data=data, modality_key = modality_key, obs_key = group_key)

	correlation_plot(data=data, modality_key = modality_key, group_key = group_key, key1 = true_key, key2 = infer_key, save = save, saving_path = saving_path, title = "True-Inferred Pseudotime Correlation", xlabel = "True Pseudotime", ylabel= "Inferred Pseudotime", xlim = (0,1 + epsilon), ylim = (0, 1 + epsilon))


def entropy_correlation(data: MuData, modality_key: Optional[str]=None, group_key: str = "celltype", entropy_key: str = "entropy", pseudo_time_key: str = "pseudotime", save:bool=True, saving_path: str= None, epsilon: float = 0.01):	

	_check_keys(data=data, modality_key = modality_key, pseudo_time_key = pseudo_time_key, entropy_key = entropy_key, obs_key = group_key)
	
	xlim = (0, 1 + epsilon)
	ylim = (0, data.obs[entropy_key].max()+ epsilon) if modality_key is None else (0, data[modality_key].obs[entropy_key].max()+ epsilon)
	correlation_plot(data, modality_key = modality_key, group_key = group_key, key1 = pseudo_time_key, key2 = entropy_key, save = save, saving_path = saving_path, title = "DP-Pseudotime Correlation", xlabel = "True Pseudotime", ylabel= "Differentiation Potential", xlim = xlim, ylim = ylim)


def correlation_plot(data:MuData, key1: str, key2:str, group_key:str, modality_key: Optional[str]=None, save:bool=True, saving_path:Optional[str]=None, **kwargs):
	
	data = data if modality_key is None else data[modality_key]

	for group in data.obs[group_key].unique():
		subdata = data[data.obs[group_key] == group]
		x = subdata.obs[key1]
		y = subdata.obs[key2]
		path = os.path.join(saving_path, f"{group}_correlation.png") if saving_path is not None else None
		simple_scatter(x=x, y=y, c=None, save=save, saving_path=path, **kwargs)
			
		


