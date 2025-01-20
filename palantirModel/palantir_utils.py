import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from palantir_environment import * 

def _check_path(path: Optional[str]=None):
	if path is None:
		path = os.path.join(os.getcwd(), "results")
	if not os.path.exists(path):
		os.mkdir(path)
	return path

class PalantirComparator:
	
	def linear_model(self, data1: Union[MuData, AnnData], data2: Union[MuData, AnnData], group_key: str, 
							data1_key: str = "palantir_fate_probabilities",
							data2_key: str = "palantir_fate_probabilities", 
							model1: str = "multiomics",
							model2: str = "rna", 
							saving_path = Optional[str] = None):
		"""
		Fits linear model and checks coefficients and statistical validity. Compares two models.
		Params:
		--------
		- data1: anndata.AnnData or muon.MuData
			First dataset to compare.
		- data2: anndata.AnnData or muon.MuData
			Second dataset to compare.
		- group_key: str
			Key in either data1.obs or data2.obs containing cell type labels. 
		- data1_key: str
			Key in data1.obsm where palantir fates probabilities are stored. Default is "palantir_fate_probabilities".
		- data2_key: str	
			Key in data2.obsm where palantir fates probabilities are stored. Default is "palantir_fate_probabilities".
		- model1: str
			String specyfing which model is data1. By default "multiomics", suggesting data1 contains results from Palantir algorithm based on multiomics data. 
		- model2: str
			String specifying which model is data2. By default "rna", suggesting data2 contains results from Palantir algorithm based on RNA data. 
		- saving_path: str, optional
			Path where results are stored. If not provided, then cwd()/results. 
		"""
		if data1_key not in data1.obsm.keys():
			raise KeyError(f"{data1_key} not in {model1}.obsm")
		if data2_key not in data2.obsm.keys():
			raise KeyError(f"{data2_key} not in {model2}.obsm")
		
		if group_key in data2.obs:
			cell_types = data2.obs[group_key]
		elif group_key in data1.obs:
			cell_types = data1.obs[group_key]
		else:
			raise KeyError(f"{group_key} not found in either {model1}.obs nor {model2}.obs")
		
		saving_path = _check_path(saving_path)	
	
		fates2 = pd.merge(data2.obsm[data2_key], cell_types, how="left", right_index=True, left_index=True) 
		fates2["model"] = [model2]*fates2.shape[0]
		fates1 = pd.merge(data1.obsm[data1_key], cell_types, how="left", right_index=True, left_index=True)
		fates1["model"] = [model1]*fates1.shape[0]
		fates = pd.concat((fates1,fates2))
		terminalStates = [cell_types.loc[ts] for ts in fates.columns[:-2]]
		fates.columns = terminalStates + list(fates.columns[-2:])		
		fates = pd.melt(fates, id_vars = ["model", group_key], var_name="terminal", value_name="probability")

		for celltype in fates[group_key].unique():
			f = fates[fates[group_key]==celltype] # Subset for a specific celltype
			model = smf.ols("probability~ model + terminal + model:terminal", data=f).fit()
			summary = model.summary()
			resultDf = pd.DataFrame(summary.tables[1].data)
			resultDf.to_csv(os.path.join(saving_path, f"{celltype}.tsv"), sep="\t")


	def plot_probability_distribution(self, data1:Union[MuData,AnnData], data2:Union[MuData,AnnData], group_key:str,
						 data1_key:str="palantir_fate_probabilities",
						 data2_key:str="palantir_fate_probabilities", 
						 model1: str = "multiomics",
						 model2: str = "rna",
						 saving_path: Optional[str] = None):
		"""
		Function that plots for each celltype the distribution of fate probabilities towards terminal states for each model. 		
		Params:
		-------
		- data1: anndata.AnnData or muon.MuData
			First dataset to compare.
		- data2: anndata.AnnData or muon.MuData
			Second dataset to compare.
		- group_key: str
			String in either data1.obs or data2.obs containing cell types.
		- data1_key: str
			Key in data1.obsm where fates probabilities are stored. Default is "palantir_fate_probabilities".
		- data2_key: str
			Key in data2.obsm where fates probabilities are stored. Default is "palantir_fate_probabilties".
		- model1: str
			String specyfing which model is data1. By default "multiomics", suggesting data1 contains results from Palantir algorithm based on multiomics data.
		- model2: str
			String specyfing which model is data1. By default "rna", suggesting data2 contains results from Palantir algorithm based on rna data.
		- saving_path: str, optional
			Path where results are stored. If not provided then cwd()/results.
		"""
		if data1_key not in data1.obsm.keys():
			raise KeyError(f"{data1_key} not in {model1}.obsm")	
		if data2_key not in data2.obsm.keys():
			raise KeyError(f"{data2_key} not in {model2}.obsm")

		if group_key in daat1.obs:
			cell_types = data1.obs[group_key]
		elif group_key in data2.obs.columns:
			cell_types = data2.obs[group_key]
		else:
			raise KeyError(f"{group_key} not found in any data.obs")

		saving_path = _check_path(saving_path)

		fates1 = pd.merge(data1.obsm[data1_key], cell_types, how="left", right_index=True, left_index=True)
		fates2 = pd.merge(data2.obsm[data2_key], cell_types, how="left", right_index=True, left_index=True)
		fates1["model"] = [model1] * len(fates1)
		fates2["model"] = [model2]*len(fates2)
		fates = pd.concat((fates1,fates2))
		terminalStates = [cell_types.loc[ts] for ts in fates.columns[:-2]]
		fates.columns = terminalStates + list(fates.columns[-2:])
		fates = pd.melt(fates, id_vars=[group_key, "model"], var_name="terminal", value_name="probability")

		for celltype in fates[group_key].unique():
			f = fates[fates[group_key]==celltype]
			g = sns.catplot(f, x="probability", y ="terminal", hue="model", kind="violin", inner="quart", split=True)
			g.fig.suptitle(f"Fate probabilities for {group_key}={celltype}")
			plt.savefig(os.path.join(saving_path, f"fateprobs_{celltype}.png"))		

