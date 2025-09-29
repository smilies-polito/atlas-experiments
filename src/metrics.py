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




def compare_transition_matrices(rna:dict, multiomics:dict):
	def l2_distance(p, q):
		p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
		n = min(len(p), len(q))
		if n == 0: return float("nan")
		p = p[:n]; q = q[:n]
		if p.sum() > 0: p = p / p.sum()
		if q.sum() > 0: q = q / q.sum()
		return float(np.linalg.norm(p - q))


	def gini(p):
		p = np.asarray(p, dtype=float)
		n = len(p)
		if n == 0: return float("nan")
		if p.sum() == 0: return float("nan")
		p = np.sort(p)
		cum = np.cumsum(p)
		g = 1 - 2 * np.sum(cum) / (n * p.sum()) + 1/n
		return float(g)


	def shannon_entropy(p):
		p = np.asarray(p, dtype=float)
		p = p[p > 0]
		return float(-(p * np.log(p)).sum()) 

	def eff_support(p):
		p = np.asarray(p, dtype=float)
		s2 = float((p**2).sum())
		return float(1.0 / s2) if s2 > 0 else float("nan")

	if not rna["stochastic"]:
		print("Rna matrix is not sotchastic")
		return
	if not multiomics["stochastic"]:
		print("Multiomics matrix is not sotchastic")
		return

	if not rna["connected"] or not rna["aperiodic"]:
		print(f"Rna matrix is connected={rna['connected']} is aperiodic={rna['aperiodic']}.\nNo stationary distribution guaranteed.")
		return
	
	if not multiomics["connected"] or not multiomics["aperiodic"]:
		print(f"Multiomics matrix is connected={multiomics['connected']} is aperiodic={multiomics['aperiodic']}.\nNo stationary distribution guaranteed.")
		return

	print(f"Number of sink components (absorbing): rna={rna['sinks']} multiomics={multiomics['sinks']}")
	print(f"Eigengap bigger in multiomics? {(multiomics['eigenGap']-rna['eigenGap'])}")
	print(f"Eigengap multiomics={multiomics['eigenGap']}, rna={rna['eigenGap']}")
	print(f"Mixing time multiomics={multiomics['mixing_time']}, rna={rna['mixing_time']}")

	multiomics_stationary_distribution_entropy = shannon_entropy(multiomics["stationary_distribution"])
	rna_stationary_distribution_entropy = shannon_entropy(rna["stationary_distribution"])
	multiomics_gini = gini(multiomics["stationary_distribution"])
	rna_gini = gini(rna["stationary_distribution"])
	multiomics_effective_support = eff_support(multiomics["stationary_distribution"])
	rna_effective_support = eff_support(rna["stationary_distribution"])
	print(f"Multiomics entropy={multiomics_stationary_distribution_entropy}, gini={multiomics_gini}, effective_support={multiomics_effective_support}")
	print(f"Rna entropy={rna_stationary_distribution_entropy}, gini={rna_gini}, effective_support={rna_effective_support}")

	print(f"L2 distance {l2_distance(multiomics['stationary_distribution'], rna['stationary_distribution'])}")


