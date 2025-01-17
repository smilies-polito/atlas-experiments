import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from muon import MuData
from anndata import AnnData
from typing import Union


class PalantirComparator:
	
	def linear_model(self, multiomics: MuData, rna: AnnData, group_key: str, 
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
		
		rnaFates = pd.merge(rna.obsm[rna_key], cell_keys, how="left", right_index=True, left_index=True) 
		rnaFates["model"] = ["rna"]*rnaFates.shape[0]
		multiomicsFates = pd.merge(multiomics.obsm[multiomics_key], cell_keys, how="left", right_index=True, left_index=True)
		multiomicsFates["model"] = ["multiomics"]*multiomicsFates.shape[0]
		fates = pd.concat((rnaFates, multiomicsFates))
		terminalStates = [cell_keys.loc[ts] for ts in fates.columns[:-2]]
		fates.columns = terminalStates + list(fates.columns[-2:])		
		fates = pd.melt(fates, id_vars = ["model", group_key], var_name="terminal", value_name="probability")

		for celltype in fates[group_key].unique():
			f = fates[fates[group_key]==celltype] # Subset for a specific celltype
			model = smf.ols("probability~ model + terminal + model:terminal", data=f).fit()
			summary = model.summary()
			resultDf = pd.DataFrame(summary.tables[1].data)
			resultDf.to_csv(f"rnaVSmultiomics_{celltype}.tsv", sep="\t")


	def plot_probability_distribution(self, multiomics:MuData, rna:AnnData, group_key:str,
						 multiomics_fates_key:str="palantir_fate_probabilities",
						 rna_fates_key:str="palantir_fate_probabilities"):
		"""
		"""
		if multiomics_fates_key not in multiomics.obsm.keys():
			raise KeyError(f"{multiomics_fates_key} not in multiomics.obsm")	
		if rna_fates_key not in rna.obsm.keys():
			raise KeyError(f"{rna_fates_key} not in rna.obsm")
		if group_key in multiomics.obs:
			cell_types = multiomics.obs[group_key]
		if group_key in rna.obs:
			cell_types = rna.obs[group_key]
		else:
			raise KeyError(f"{group_key} not found in any data.obs")

		multiomicsFates = pd.merge(multiomics.obsm[multiomics_fates_key], cell_type, how="left", right_index=True, left_index=True)
		rnaFates = pd.merge(rna.obs[rna_fates_key], cell_type, how="left", right_index=True, left_index=True)
		multiomicsFates["model"] = ["multiomics"] * len(multiomicsFates)
		rnaFates["model"] = ["rna"]*len(rnaFates)
		fates = pd.concat((rnaFates, multiomicsFates))
		terminalStates = [data.obs[group_key].loc[ts] for ts in fates.columns[:-2]]
		fates.columns = terminalStates + list(fates.columns[-2:])
		fates = pd.melt(fates, id_vars=[group_key, "model"], var_name="terminal", value_name="probability")
		plots = []		

		for cell_type in fates[group_key].unique():
			f = fates[fates[group_key]==cell_type]
			plots.append(sns.catplot(f, x="probability", y ="terminal", hue="model", kind="violin", inner="quart", split=True))
		
		return plots		

