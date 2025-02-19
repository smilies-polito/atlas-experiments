import os
import numpy as np
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
from muon import MuData
from .palantir_utils import _check_keys
from anndata import AnnData
from typing import Union, Optional

def plot_palantir_results(data: Union[AnnData, MuData], modality_key: Optional[str]= None,
							embedding_key: str = "X_umap",
							pseudo_time_key: str = "palantir_pseudotime",
							entropy_key: Optional[Union[str, list]]= "palantir_entropy",
							fate_prob_key: str = "palantir_fate_probabilities",
							save:bool = True,
							saving_path: Optional[str]=None):
	"""
	"""
	if isinstance(entropy_key, str):
		entropy_key = [entropy_key]
	if modality_key is None:
		print("No modality has been specified, using MuData Palantir run.")
		entropy_key = _check_keys(data, embedding_key=embedding_key, pseudo_time_key = pseudo_time_key, entropy_key=entropy_key,
				fate_prob_key = fate_prob_key)
		data = data
		embedding = data.obsm[embedding_key] 
	else:
		if modality_key not in data.mod.keys():
			raise KeyError(f"Modality {modality_key} not in data.mod")
		print(f"Modality {modality_key} specified, using data[modality_key] Palantir run.")
		entropy_key = _check_keys(data, modality_key=modality_key, embedding_key=embedding_key, pseudo_time_key=pseudo_time_key, 
				entropy_key = entropy_key, fate_prob_key = fate_prob_key)
		embedding = data.obsm[embedding_key]
		data = data[modality_key]

	plot_probabilities(data, embedding, fate_prob_key=fate_prob_key, save=save, saving_path = saving_path)
	plot_entropy(data, embedding, entropy_key=entropy_key, save=save, saving_path = saving_path)
	plot_pseudotime(data, embedding, pseudo_time_key = pseudo_time_key, save=save, saving_path=saving_path)


def plot_probabilities(data: Union[AnnData, MuData], embedding: np.ndarray, fate_prob_key: str = "palantir_fate_probabilities",
						save:bool = True, saving_path: Optional[str]=None):
	"""
	"""
	_ = _check_keys(data, fate_prob_key = fate_prob_key)
	if save:
		if saving_path is not None and not os.path.exists(saving_path):
			os.mkdir(saving_path) 
		if saving_path is None:
			os.mkdir(os.path.join(os.getcwd(), "palantir_results"))
		
	
	df = data.obsm[fate_prob_key].copy()
	terminal_states = df.columns
	df["x"] = embedding[:,0]
	df["y"] = embedding[:,1]
	if len(terminal_states)==0:
		warnings.warn("No terminal states detected!")
		simple_scatter(x=df.x, y=df.y, c=np.zeros(len(df)), cmap="GnBu", title=fate_prob_key, saving_path= saving_path)
		return

	for terminal in terminal_states:
		plt.figure(figsize=(8,6))
		scatter = plt.scatter(df["x"], df["y"], c=df[terminal], cmap="GnBu")
		terminal_x, terminal_y = df.loc[terminal, "x"], df.loc[terminal, "y"] 
		plt.scatter(terminal_x, terminal_y, color="red")
		plt.xticks([])
		plt.yticks([])
		if save:
			plt.savefig(os.path.join(saving_path, f"{terminal}_fates.png"))
		plt.close()
	

def plot_entropy(data: MuData, embedding: np.ndarray, entropy_key: Union[str, list] = "palantir_entropy", 
					save:bool = True, saving_path:Optional[str]=None ):
	keys = _check_keys(data, entropy_key=entropy_key)
	df = data.obs[keys].copy()
	df["x"] = embedding[:,0]
	df["y"] = embedding[:,1]
	
	if save:
		if saving_path is not None and not os.path.exists(saving_path):
			os.mkdir(saving_path)
		elif saving_path is None:
			os.mkdir(os.path.join(os.getcwd(), "palantir_results"))
	for k in keys:
		simple_scatter(x=df.x, y=df.y, c=df[k], title=k, save=save, saving_path=saving_path)


def plot_pseudotime(data:MuData, embedding: np.ndarray, pseudo_time_key:str="palantir_psedotime", save:bool=True, 
				saving_path:Optional[str]=None):
	_ = _check_keys(data, pseudo_time_key = pseudo_time_key)
	df = data.obs[pseudo_time_key].copy()
	df["x"] = embedding[:,0]
	df["y"] = embedding[:,1]
	df[pseudo_time_key]=data.obs[pseudo_time_key]
	
	if save:
		if saving_path is not None and not os.path.exists(saving_path):
			os.mkdir(saving_path)
		elif saving_path is None:
			os.mkdir(os.path.join(os.getcwd(), "palantir_results"))

	simple_scatter(x = df.x, y=df.y, c=df[pseudo_time_key], title=pseudo_time_key, save=save, saving_path=saving_path)



def simple_scatter(x, y, c, cmap:Optional[str]=None, title:Optional[str]=None, save:bool=True, saving_path:Optional[str]=None):
		plt.figure(figsize=(8,6))
		cmap = cmap if cmap is not None else "YlOrBr"
		scatter= plt.scatter(x, y, c=c, cmap=cmap)
		plt.xticks([])
		plt.yticks([])
		plt.title(title)
		plt.colorbar(scatter)
		if save:
			plt.savefig(os.path.join(saving_path, f"{title}.png"))
		plt.close()
		
		
