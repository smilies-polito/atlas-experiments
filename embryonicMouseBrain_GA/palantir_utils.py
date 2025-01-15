import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from muon import MuData
from anndata import AnnData
from typing import Union
import seaborn as sns

class PalantirComparator:
	
	def plot_avg_probability(self, multiomics: MuData, rna: AnnData, group_key: str, 
							multiomics_key: str = "palantir_fate_probabilities",
							rna_key: str = "palantir_fate_probabilities"):
		"""
		"""
		if multiomics_key not in multiomics.obsm.keys():
			raise KeyError(f"{multiomics_key} not in multiomics.obsm")
		if rna_key not in rna.obsm.keys():
			raise KeyError(f"{rna_key} not in rna.obsm")
		
		if group_key in rna.obs:
			cell_keys = rna.obs[group_key]
		elif group_key in multiomics.obs:
			cell_keys = multiomics.obs[group_key]
		else:
			raise KeyError(f"{group_key} not found in either rna.obs nor multiomics.obs")
		
		rna_df = pd.merge(rna.obsm[rna_key], cell_keys, how="left", right_index=True, left_index=True) 
		rna_df["type"] = ["rna"]*rna_df.shape[0]
		multiomics_df = pd.merge(multiomics.obsm[multiomics_key], cell_keys, how="left", right_index=True, left_index=True)
		multiomics_df["type"] = ["multiomics"]*multiomics_df.shape[0]
		fates_df = pd.concat((rna_df, multiomics_df))
		avg_fates = fates_df.groupby(["type", group_key]).mean()
		avg_fates = avg_fates.reset_index()
	
		return sns.pointplot(avg_fates, x="rna:celltype", y=avg_fates.columns[2], hue="type", linestyle="none")
