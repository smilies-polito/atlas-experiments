import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from anndata import AnnData
from muon import MuData
from typing import Optional, Union

def _check_keys(data: Union[AnnData, MuData], modality_key:Optional[str]=None, 
				embedding_key: Optional[str]=None, pseudo_time_key: Optional[str]=None,
				entropy_key: Optional[str]=None, fate_prob_key: Optional[str]=None, obs_key:Optional[str]=None):

	is_mudata = isinstance(data, MuData) and modality_key is None
	is_anndata = isinstance(data, AnnData)
	is_modality = isinstance(data, MuData) and modality_key is not None
	if is_modality and modality_key not in data.mod.keys():
		raise KeyError(f"{modality_key} not in data.mod.keys()")
	if (is_mudata or is_modality) and embedding_key is not None and embedding_key not in data.obsm.keys():
 		raise KeyError(f"{embedding_key} not in data.obsm") 
	if (is_mudata or is_anndata) and (pseudo_time_key is not None and pseudo_time_key not in data.obs.columns):
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if is_modality and (pseudo_time_key is not None and pseudo_time_key not in data[modality_key].obs.columns):
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if (is_mudata or is_anndata) and (fate_prob_key is not None and fate_prob_key not in data.obsm.keys()):
		raise KeyError(f"{fate_prob_key} not in data.obsm")
	if is_modality and (fate_prob_key is not None and fate_prob_key not in data[modality_key].obsm.keys()): 
		raise KeyError(f"{fate_prob_key} not in data.obsm") 
	if (is_mudata or is_anndata) and (entropy_key is not None and entropy_key not in data.obs.columns):
		raise KeyError(f"{entropy_key} not in data.obs")
	if is_modality and entropy_key is not None and entropy_key not in data[modality_key].obs.columns:
		raise KeyError(f"{entropy_key} not in data[{modality_key}].obs")
	if (is_anndata or is_mudata) and obs_key is not None and obs_key not in data.obs.columns:
		raise KeyError(f"{obs_key} not in data.obs")
	if is_modality and obs_key is not None and obs_key not in data[modality_key].obs.columns:
		raise KeyError(f"{obs_key} not in data[{modality_key}].obs")

def _save_results(data: MuData, entropy_key:str = "entropy", pseudo_time_key: str = "palantir_entropy", fate_prob_key: str = "fates", modality_key: str = None, **kwargs):
	group_key = kwargs["group_key"] if "group_key" in kwargs else None
	_check_keys(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key=pseudo_time_key, obs_key=group_key, fate_prob_key = fate_prob_key)
	
	true_key = kwargs["true_pseudotime"] if "true_pseudotime" in kwargs else None
	_check_keys(data, modality_key = modality_key, obs_key = true_key)
		
	data = data if modality_key is None else data[modality_key]
	model = "multiomics" if modality_key is None else modality_key 

	columns = [entropy_key, pseudo_time_key]
	if group_key is not None:
		columns = columns + [group_key]
	if true_key is not None:
		columns = columns + [true_key]	
	dataframe = data.obs[columns].copy()
	dataframe["model"] = model
	dataframe = pd.concat((dataframe, data.obsm[fate_prob_key]), axis=1)

	return dataframe
