import numpy as np
import pandas as pd
from typing import Literal
from cellrank.estimators import GPCCA
from sklearn.metrics.pairwise import paired_distances
from scipy.stats import pearsonr, spearmanr, kendalltau


def _check_macrostate_quality(g:GPCCA, ns:int):
	minChi = np.min(g.macrostates_memberships.X)
	crispness = g._gpcca.crispness_values[0]
	return {"minChi":minChi, "crispness":crispness}

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
