import os
import numpy as np
import pandas as pd
from typing import Literal
from cellrank.estimators import GPCCA
from core.utils import simple_scatter, simple_heatmap
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import paired_distances


def _check_macrostate_quality(g:GPCCA, ns:int):
	minChi = np.min(g.macrostates_memberships.X)
	crispness = g._gpcca.crispness_values[0]
	return {"minChi":minChi, "crispness":crispness}


def plot_branch_correlation(dataframe: pd.DataFrame, key1:str, key2:str, group_key:str, color_key:str, branch:dict, save:bool=True, saving_path:str = None, **kwargs):
	if key1 not in dataframe.columns:
		raise KeyError(f"{key1} not in dataframe.columns")
	if key2 not in dataframe.columns:
		raise KeyError(f"{key2} not in dataframe.columns")
	if group_key not in dataframe.columns:
		raise KeyError(f"{group_key} not in dataframe.columns")
	if color_key not in dataframe.columns:
		raise KeyError(f"{color_key} not in dataframe.columns")
	title = kwargs.get("title", "")

	for k, v in branch.items():
		subsetdata = dataframe[dataframe[group_key].isin(v)]
		path = os.path.join(saving_path, f"{k}_branch_scatter.png") if saving_path is not None else None
		kwargs["title"] = title + f" {k}" 
		simple_scatter(x=subsetdata[key1], y=subsetdata[key2], c=subsetdata[color_key], save=save, saving_path=path, categorical=True, **kwargs)


	
def compute_correlation(group, method_key: Literal["pearson", "kendall-tau", "spearman"], key1:str, key2:str):
	if method_key == "pearson":
		method = pearsonr
	elif method_key == "kendall-tau":
		method = kendalltau 
	else:
		method = spearmanr

	corr,pvalue = method(group[key1], group[key2])
	return(corr, pvalue)


def geometric_mean(values:np.array):
	return np.exp(np.mean(np.log(values)))

def compute_f1(inferred:np.array, truth:np.array, aggregate:bool=False):
	cosine_distance = paired_distances(inferred, truth, metric="cosine")[0]
	euclidean_distance = paired_distances(inferred, truth, metric = "euclidean")[0]
	if aggregate:
		return (geometric_mean(cosine_distance), geometric_mean(euclidean_distance))
	else:	
		return (cosine_distance, euclidean_distance)	
