import os
import warnings
import numpy as np 
import pandas as pd
from typing import Optional
from collections.abc import Callable
from scipy.spatial.distance import cdist, jensenshannon
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr, permutation_test, bootstrap, kendalltau


def pearson_entropy_pseudotime(entropy: pd.Series, pseudotime: pd.Series, seed:int=42, n_resamples:int=10000,
								confidence_level: float=0.95) -> tuple:
	'''
	Pearson Correlation between entropy and pseudotime. Measures the linear correlation between entropy and pseudotime. 
	Motivation: pluripotency measurement (entropy) descreases/increases according to pseudotime. Less developed cells have low pseudotime and high/low entropy according to algorithm and entropy measurement. 
	Parameters:
	- entropy: pandas.Series; contains entropy values for each cell. 
	- pseudotime: pandas.Series; contains pseudotime values for each cell. 
	- seed: int; seed for resampling. Defalut is 42.
	- n_resamples: int; number of resamles for permutation_test, bootstrap. Default is 10000.
	- confidence_level: float; confidence level for bootstrap. Default is 0.95. 
	Output:
	- tuple containing correlation and associated pvalue.	

	Notes:
	------
	When the correlation is exactly ±1, the bootstrap distribution is degenerate and confidence intervals may be NaN.
	'''
	def stat(x:np.ndarray, y:np.ndarray) -> float: 
		return pearsonr(x,y).statistic

	if len(entropy)!=len(pseudotime):
		raise ValueError(f"#instances do not match {len(pseudotime)} != {len(entropy)}")

	if entropy.isna().any() or pseudotime.isna().any():
		raise ValueError(f"NaNs are not a valid input")

	entropy = entropy.reindex(pseudotime.index)
	# Exact p-value with permutation and confidence interval via bootstrapping
	if len(entropy) < 500:
		warnings.warn("Less than 500 samples, pvalue might not be accurate. Permutation test and bootstrapping.")
		res = permutation_test((pseudotime.values, entropy.values), 
								stat, 
								permutation_type="pairings", 
								n_resamples=n_resamples, 
								alternative="two-sided", 
								random_state=seed)
		ci = bootstrap((pseudotime.values, entropy.values),
						stat,
						n_resamples = n_resamples, 
						paired = True,
						alternative = "two-sided",
						confidence_level= confidence_level,
						random_state= seed)
						 
	else:
	 	res, ci = pearsonr(pseudotime.values, entropy.values), {}
	return (res.statistic, res.pvalue, getattr(ci, "confidence_interval", None))

	

def spearman_entropy_pseudotime(entropy: pd.Series, pseudotime: pd.Series, seed:int=42, n_resamples:int=10000,
								confidence_level:float=0.95) -> tuple:
	'''
	Spearman Correlation between entropy and pseudotime. Measures the monotone correlation between entropy and pseudotime. 
	Motivation: pluripotency measurement (entropy) descreases/increases according to pseudotime. Less developed cells have low pseudotime and high/low entropy according to algorithm and entropy measurement. 
	Parameters:
	- entropy: pandas.Series; contains entropy values for each cell. 
	- pseudotime: pandas.Series; contains pseudotime values for each cell. 
	- seed: int; seed for resampling. Default is 42.
	- n_resamples: int; number of resamles for permutation_test, bootstrap. Default is 10000.
	- confidence_level: float; confidence level for bootstrap. Default is 0.95. 
	Output:
	- tuple containing correlation and associated pvalue.

	Notes:
	------
	When the correlation is exactly ±1, the bootstrap distribution is degenerate and confidence intervals may be NaN.
	'''
	def stat(x:np.ndarray, y:np.ndarray) -> float: 
		return spearmanr(x,y).statistic

	if len(entropy)!=len(pseudotime):
		raise ValueError(f"#instances do not match {len(pseudotime)} != {len(entropy)}")

	if entropy.isna().any() or pseudotime.isna().any():
		raise ValueError(f"NaNs are not a valid input")

	entropy = entropy.reindex(pseudotime.index)
	if len(entropy) < 500:
		warnings.warn("Less than 500 samples, pvalue might not be accurate. Consider bootstrapping")
		res = permutation_test((pseudotime.values, entropy.values), 
								stat, 
								permutation_type="pairings", 
								n_resamples= n_resamples,
								alternative="two-sided", 
								random_state=seed)
		ci = bootstrap((pseudotime.values, entropy.values),
						stat,
						n_resamples = n_resamples, 
						paired = True,
						alternative = "two-sided",
						confidence_level= confidence_level,
						random_state= seed)
	else:
		res, ci  = spearmanr(pseudotime.values, entropy.values), {}
	return (res.statistic, res.pvalue, getattr(ci, "confidence_interval", None))



def fate_concentration_index(fates:pd.DataFrame, pseudotime:pd.Series, seed:int = 42, n_resamples:int=10000, confidence_level:float=0.95) -> tuple:
	'''
	Fate Concentration Index correlation between fate concentration and pseudotime. Measures the monotone increase of fate probability concentration in one lineage and pseudotime. Motivation relies in the fact that cells with higher pseudotime should be more committed to a specific lineage. 
	Parameters:
	- fates: pandas.DataFrame; Shape is n_cells x n_terminal states and each element contains the probability of the i-th cell to commit to the t-th terminal state. 
	- pseudotime: pandas.Series; pseudotime for each cell. 
	- seed: int; seed for resampling. Default is 42.
	- n_resamples: int; number of resamles for permutation_test, bootstrap. Default is 10000.
	- confidence_level: float; confidence level for bootstrap. Default is 0.95. 
	Output:
	- tuple containing correlation and associated pvalue
	'''
	def stat(x:np.ndarray, y:np.ndarray) -> float: 
		return spearmanr(x,y).statistic

	def _concentration_index(x:pd.DataFrame) -> pd.Series:
		'''
		Simpson concentration index. For each cell it computed the concentration index c_i = sum_t(f_it**2).
		The higher the index the more "concentrated" a fate is towards a specific lineage.
		Parameters:
		- x; pd.DataFrame; Shape is NxT.
		Output:	
		- ci: pd.Series; Shape is N.
		'''
		return (x**2).sum(axis=1)

	if fates.shape[1] == 0:	
		raise ValueError(f"No terminal probabilities are found. Provide a 2D dataframe n_cells x n_terminal_states")
	if len(pseudotime) != fates.shape[0]:
		raise ValueError(f"#instances do not match {len(pseudotime)} != {fates.shape[0]}")
	
	pseudotime = pseudotime.reindex(index=fates.index)
	concentration_index = _concentration_index(fates)

	if len(pseudotime) < 500:
		warnings.warn("Less than 500 samples, pvalue might not be accurate. Consider bootstrapping")
		res = permutation_test((pseudotime.values, concentration_index.values), 
								stat, 
								permutation_type="pairings", 
								n_resamples= n_resamples,
								alternative="two-sided", 
								random_state=seed)
		ci = bootstrap((pseudotime.values, concentration_index.values),
						stat,
						n_resamples = n_resamples, 
						paired = True,
						alternative = "two-sided",
						confidence_level= confidence_level,
						random_state= seed)
	else:
		res, ci  = spearmanr(pseudotime.values, concentration_index.values), {}
	return (res.statistic, res.pvalue, getattr(ci, "confidence_interval", None), concentration_index)
	

def terminal_state_silhouette(fates: pd.DataFrame, 
								soft_assignment:bool=False,
								pseudotime: Optional[pd.Series] = None,
								confidence_filtering:bool=False, 
								pseudotime_weight:bool=False, 
								confidence_parameter:float=0.6, 
								alpha:float = 1) -> tuple:
	'''
	Terminal state silhouette investigates whether terminal states are well separated. This metric investigates if
	cells committed towards a lineage are more similar between them then with cells belonging to other lineages.
	Parameters:
	- fates: pandas.DataFrame; Dataframe of shape n_cells x n_terminal_states where each element is the probability
	of the cell to develop towards that state. 		
	- soft_assignemnt: bool; Indicates wether to deal with pluripotent cells according to the soft assignment strategy (see later).
	- pseudotime: optional, pandas.Series; Series of shape n_cells containing pseudotime values for each cell. Default is None.
	- confidence_filtering: bool; Indicates whether to deal with pluripotent cells according to the confidence based 
	filtering strategy (see later). Default is False. 
	- pseudotime_weight: bool; Indicates whether to deal with pluritpotent cells according to the pseudotime weighted 
	strategy (see later). Default is True.
	Default is False.
	- confidence_parameter: float; Indicates the confidence theta used in confidence based filtering strategy.
	Default is 0.6. 
	- alpha: float; Penalization parameter for pseudotime weight. Default is 1. 

	Silhouette score for terminal states might be biased for non-fully-committed cell types. Some strategies can be	
	adopted to deal with this scenario:
	- Confidence Based Filtering Strategy: each cell is assigned to a lineage using most probable fate strategy argmax(p_it).
		Computes silhouette only considering those cells with argmax(p_it) > theta.
	- Pseudotime Weight Strategy: the final silhouette metric is weighted according to pseudotime. S = sum_i(tau_i^alpha * s_i) / sum_i(tau_i^alpha)
	- Soft Assignment: computes silhouette not assigning each cell to a specific lineage. It helps evaluating how each cell shares its fate
	with other cells.
	'''

	def _bi(f: pd.DataFrame, soft_strategy:bool=True) -> pd.Series:
		'''
		Computes b_i for each cell. B_i defines the nearest terminal distance.
		Parameters: 
		- f: pandas.DataFrame;
		- soft_strategy: bool; Whether to adopt a soft strategy (default) or hard strategy.
		Output: b_i value for each cell. 

		hard_case_scenario: b_i quantifies the average distance between the i-th cell and cells assigned to the closest aternative terminal state. It measures the separation of the terminal predicted state and all others. 
		soft case scenatio: b_i quantifies the average sitance betenn the i-th cell and all other cells, weighted by the probabiliy of belonging to an alternative terminal fate. It measures the continuous separation of cells from different terminal outcomes.
		'''
		cell_ids, terminal = f.index, f.columns
		D = pd.DataFrame(cdist(f.values, f.values, metric="euclidean"),
							index = cell_ids, 
							columns = cell_ids)
		bi = pd.Series(index=cell_ids, dtype=float)

		if soft_strategy:
			for i in cell_ids:
				p_same = f.loc[i].values @ f.values.T
				p_same[f.index.get_loc(i)] = 0 
				p_alt = 1.0 - p_same
				p_alt[f.index.get_loc(i)] = 0
				bi[i] = np.nan if p_alt.sum() == 0 else np.sum(p_alt * D.loc[i].values)/np.sum(p_alt)
		else:
			assignment = f.idxmax(axis="columns")
			for i in cell_ids:
				ti = assignment.loc[i]
				b_candidates = []
				for t in terminal:
					if t == ti:
						continue
					cells_t = assignment[assignment==t].index
					if len(cells_t) == 0:
						continue
					b_t = D.loc[i, cells_t].mean()
					b_candidates.append(b_t)
				bi[i] = np.nan if len(b_candidates) == 0 else np.min(b_candidates)

		return bi					


	def _ai(f: pd.DataFrame, soft_strategy:bool=True) -> pd.Series:
		'''
		Computes a_i for each cell. A_i defines the mean intra-terminal distance.
		Parameters: 
		- f: pandas.DataFrame;
		- soft_strategy: bool; Whether to adopt a soft strategy (default) or hard strategy.
		Output: a_i value for each cell. 

		hard case scenario: a_i quantifies the average distance between the i-th cell and all other cells assigned to the same terminal state. It measures the internal cohesion of the predicted terminal state to which the i-th cell belongs. 
		soft case scenario: a_i quantifies the average distance between the i-th cell and all other cells weighted by the probability that such cells share the same terminal fate. It measures the cohesion among cells with similar fate probabilities. 
		'''
		cell_ids = f.index
		D = pd.DataFrame(cdist(f.values, f.values, metric="euclidean"),
							index = cell_ids, 
							columns = cell_ids)
		ai = pd.Series(index=cell_ids, dtype=float)

		if soft_strategy:
			for i in cell_ids:
				w = f.loc[i].values @ f.values.T
				w[f.index.get_loc(i)] = 0
				ai[i] = np.nan if w.sum() == 0 else np.sum(w*D.loc[i].values)/np.sum(w)

		else:
			assignment = f.idxmax(axis="columns")
			for i in cell_ids:
				t = assignment.loc[i]
				same_terminal = assignment[assignment==t].index
				same_terminal = same_terminal.drop(i)
				ai[i] = np.nan if len(same_terminal)==0 else D.loc[i, same_terminal].mean()
		return ai 

	if fates.shape[1] == 1:
		warnings.warn("Single terminal state")
		return(0,-1) 

	if not soft_assignment:
		if pseudotime_weight and confidence_filtering:
			raise ValueError("Only one strategy among soft_assignment, pseudotime_weight and confidence filtering must be specified")
		if not pseudotime_weight and not confidence_filtering:
			raise ValueError("At least one strategy among soft_assignment, pseudotime_weight and confidence filtering must be specified")
		if pseudotime_weight and pseudotime is None:
			raise ValueError(f"Pseudotime weight strategy requires pseudotime for all cells to be provided")
	if soft_assignment:
			pseudotime_weight, confidence_filtering = False, False 

	ai = _ai(f=fates, soft_strategy=True) if soft_assignment else _ai(f=fates, soft_strategy=False)
	bi = _bi(f=fates, soft_strategy = True) if soft_assignment else _bi(f=fates, soft_strategy=False)
	si = (bi - ai) / np.maximum(ai, bi)
	if soft_assignment:
		S = np.mean(si)
		return (S, None)

	if confidence_filtering:
		committed_group = np.max(fates, axis=1) > confidence_parameter
		if np.sum(committed_group) == 0:
			warnings.warn("No cell seems to be committed to a specific fate.")
		S = np.mean(si[committed_group]) 
		return (S, np.sum(committed_group))

	if pseudotime_weight: 
		pseudotime = pseudotime.reindex(fates.index)
		S = np.sum((pseudotime**alpha)*si)/np.sum(pseudotime**alpha)
		return (S, None)
	

