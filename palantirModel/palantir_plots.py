import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
from muon import MuData
from scipy.stats import pearsonr
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
		simple_scatter(x=df.x, y=df.y, c=np.zeros(len(df)), cmap="GnBu", title=fate_prob_key, saving_path= os.path.join(saving_path, "no_probs.png"))
		return

	for terminal in terminal_states:
		plt.figure(figsize=(8,6))
		scatter = plt.scatter(df["x"], df["y"], c=df[terminal], cmap="GnBu")
		terminal_x, terminal_y = df.loc[terminal, "x"], df.loc[terminal, "y"] 
		plt.scatter(terminal_x, terminal_y, color="red")
		plt.xticks([])
		plt.yticks([])
		plt.title(f"Fates towards {terminal}")
		plt.colorbar(scatter)
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
			saving_path = os.path.join(os.getcwd(), "palantir_results")
	for k in keys:
		simple_scatter(x=df.x, y=df.y, c=df[k], title=f"{k}", save=save, saving_path=os.path.join(saving_path, f"{k}.png"), xticks = None, xlim=None,
			xlabel = None, yticks= None, ylim=None, ylabel=None)


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
			saving_path = os.path.join(os.getcwd(), "palantir_results")

	simple_scatter(x = df.x, y=df.y, c=df[pseudo_time_key], title= "Palantir Pseudotime", save=save, saving_path = os.path.join(saving_path, "pseudotime.png"),
				xticks = None, xlim=None, yticks=[], ylim = None, xlabel = None, ylabel=None) 


def correlation_plot(data:MuData, modality_key: Optional[str]=None, key1: str= "pseudotime", key2:str="entropy", group_key:Optional[str]="celltype", save:bool=True, saving_path:Optional[str]=None):
	
	if modality_key is not None and modality_key not in data.mod.keys():
		raise KeyError(f"{modality_key} not in data.obs")
	adata = data if modality_key is None else data[modality_key]
	if key1 not in adata.obs.columns:
		raise KeyError(f"{key1} not in data.obs.columns")
	if key2 not in adata.obs.columns:
		raise KeyError(f"{key2} not in data.obs.columns")
	if group_key is not None and group_key not in adata.obs.columns:
		raise KeyError(f"{group_key} not in data.obs.columns")
	if group_key is not None:	
		correlations = {}
		for group in adata.obs[group_key].unique():
			subdata = adata[adata.obs[group_key] == group]
			x = subdata.obs[key1]
			y = subdata.obs[key2]
			simple_scatter(x,y,c=None, cmap=None, title=f"{group} correlation", 
				saving_path = os.path.join(saving_path, f"{group}_correlation.png"), save = save, 
				xticks = np.linspace(0,1,10), xlim=(0,1.1), xlabel = key1, ylabel= key2)
			corr = pearsonr(x, y)
			correlations[group] = (corr.statistic, corr.pvalue)
		pd.DataFrame(correlations).to_csv(os.path.join(saving_path, "correlations.tsv"), sep="\t", header=True, index=True)
			
	else:
		x = adata.obs[key1]
		y = adata.obs[key2]
		simple_scatter(x,y,c=None, title = "correlation", save = save, 
						saving_path = os.path.join(saving_path, f"{key1}{key2}_correlation.png"), 
						xticks = np.linspace(0,1,10), xlim = (0, 1.1), xlabel = key1, ylabel=key2)
		




def simple_scatter(x, y, c:Optional[Union[list, np.ndarray, pd.Series]] = None, cmap:Optional[str]="YlOrBr", title:Optional[str]=None, 
					save:bool=True, saving_path:Optional[str]=None, xticks: Optional[np.ndarray]=None, yticks: Optional[np.ndarray]=None,
					ylim: Optional[tuple] = None, xlim:Optional[tuple] = None, xlabel:Optional[str] = None, ylabel:Optional[str]=None):
	plt.figure(figsize=(8,6))

	if cmap is not None:
		scatter= plt.scatter(x, y, c=c, cmap=cmap)
		plt.colorbar(scatter)
	else:
		scatter = plt.scatter(x,y)

	if ylim is not None:
		plt.ylim(ylim[0], ylim[1])
	if xlim is not None:
		plt.xlim(xlim[0], xlim[1]) 

	if xticks is not None:
		plt.xticks(xticks)
	if yticks is not None:
		plt.yticks(yticks)
	
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)

	if title is not None:
		plt.title(title)
	
	if save:
		plt.savefig(saving_path)
	plt.close()
		

