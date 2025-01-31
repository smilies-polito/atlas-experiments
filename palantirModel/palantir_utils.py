import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from .palantir_environment import * 

def _check_keys(data: Union[AnnData, MuData], modality_key:Optional[str]=None, 
				embedding_key: Optional[str]=None, pseudo_time_key: Optional[str]=None,
				entropy_key: Optional[Union[str, list]]=None, fate_prob_key: Optional[str]=None):
	is_mudata = isinstance(data, Mudata) and modality_key is None
	is_anndata = isinstance(data, AnnData)
	is_modality = isinstance(data, Mudata) and modality_key is None

	if is_modality and modality_key not in data.mod.keys():
		raise KeyError(f"{modality_key} not in data.mod.keys()")
	if is_mudata and embedding_key is not None and embedding_key not in data.obsm.keys():
 		raise KeyError(f"{embedding_key} nopt in data.obsm") 
	if is_mudata or is_anndata and pseudo_time_key not in data.obs.columns:
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if is_modality and pseudo_time_key not in data[modality_key].obs.columns:
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if is_mudata or is_anndata and entropy_key not in data.obsm.keys():
		raise KeyError(f"{fate_prob_key} not in data.obsm")
	if is_modality and fate_probs_key not in data[modality_key].obsm.keys(): 
		raise KeyError(f"{fate_prob_key} not in data.obsm") 
	if isinstance(entropy_key, str):  
		entropy_key = [entropy_key]
	if is_mudata or is_anndata: 
		entropy_key = [key for key in entropy_key if key in data.obs.columns] 
	if is_modality:
		 entropy_key = [key for key in entropy_key if key in data[modality_key].obs.columns]  
	return entropy_key

class PalantirComparator:
	def linear_model(self, data1: Union[MuData, AnnData], data2: Union[MuData, AnnData],
							group_key: str, 
							is_fate:bool = True,
							key1: str = "palantir_fate_probabilities",
							key2: str = "palantir_fate_probabilities", 
							model1: str = "multiomics",
							model2: str = "rna", 
							save: bool= False, saving_path: Optional[str] = None):
		"""
		Fits linear model on fate probability comparing its dependence on model type (e.g., multiomics, rna, etc.) and terminal state for every celltype.
		Params:
		--------
		- data1: anndata.AnnData or muon.MuData
			First dataset to compare.
		- data2: anndata.AnnData or muon.MuData
			Second dataset to compare.
		- group_key: str
			Key in either data1.obs or data2.obs containing cell type labels. 
		- is_fate: str
			Boolean indicated whether to fit the model for fates probabilities or for entropy. Default is True.
		- key1: str
			Key in data1.obsm where palantir fates probabilities are stored if is_fate is True, otherwise key in data1.obs where entropy values are stored. Default is "palantir_fate_probabilities".
		- key2: str	
			Key in data2.obsm where palantir fates probabilities are stored if if_fate is True, otherwise kei in data2.obs where entropy values are stored. Default is "palantir_fate_probabilities".
		- model1: str
			String specyfing which model is data1. By default "multiomics", suggesting data1 contains results from Palantir algorithm based on multiomics data. 
		- model2: str
			String specifying which model is data2. By default "rna", suggesting data2 contains results from Palantir algorithm based on RNA data. 
		- saving: bool
			Whether to save results in saving_path. Default is True.
		- saving_path: str, optional
			Path where results are stored. If not provided, then cwd()/results. 
		"""
		if is_fate:
			lm = FatesLinearModel(data1, data2, group_key, key1=key1, key2=key2, model1=model1, model2=model2, save=save, saving_path=saving_path)
		else: 
			lm = EntropyLinearModel(data1, data2, group_key, key1=key1, key2=key2, model1=model1, model2=model2, save=save, saving_path=saving_path)
		lm.fit()	

	def save_palantir_matrix(self, data1:Union[MuData, AnnData], data2: Union[MuData, AnnData], 
							group_key:str, is_fate:bool=True,  
							key1:str = "palantir_fate_probabilities", key2: str= "palantir_fate_probabilities", 
							model1: str= "multiomics", model2:str = "rna",
							saving_path: Optional[str] = None):
		"""
		Function that saves palantir matrix for linear model as .tsv.
		Params:
		--------
		- data1: anndata.AnnData or muon.MuData
		- data2: anndata.AnnData or muon.MuData
		- group_key: str
			Key in either data1.obs or data2.obs identifying cell types.
		- is_fate: boolean
			Indicates whether fate probabilities or entropy is investigated. Default is True (meaning fate probs).
		- key1: str
			Key in data1.obsm where fate probabilities are stored if is_fate is True, else key in data.obs where entropy values are stored. Default is "palantir_fate_probabilities".
		- key2: str
			Key in data2.obs, where fate probabilities are stored in if_fate is True, else key in data.obs where entropy values are stored. Default is "palantir_fate_probabilities".
		- model1: str
			String identifying which model is data1. Default is "multiomics", suggesting data1 contains results of Palantir algorithm based on multiomics data.
		- model2: str
			String identifying which model is data2. Default is "rna", suggesting data2 contains results of Palantir algorithms based on rna data only. 
		_ saving_path: str, optional
			Path where to store the results. If not provided, then cwd()/results/model.tsv is used.
		"""
		if is_fate:
			lm = FatesLinearModel(data1, data2, group_key, key1=key1, key2=key2, model1=model1, model2=model2, save=False)
		else: 
			lm = EntropyLinearModel(data1, data2, group_key, key1=key1, key2=key2, model1=model1, model2=model2, save= False)

		if saving_path is None:
			saving_path = os.path.join(os.getcwd(), "results", "model.tsv")
		lm.data.to_csv(saving_path, sep="\t", header=True, index=False)

		

	def plot_entropy_distribution(self, data1:Union[MuData, AnnData], data2:Union[MuData, AnnData], group_key:str,
						data1_key: str = "palantir_entropy", data2_key: str= "palantir_entropy", 
						model1: str = "multiomics", model2: str = "rna", 
						save: bool = True, saving_path: Optional[str]=None):
		""" 
		Function that plots for each cell type the entropy distribution for each model.
		Params:
		-------
		"""
		if data1_key not in data1.obs.columns:
			raise KeyError(f"{data1_key} not in {model1}.obs")
		if data2_key not in data2.obs.columns:
			raise KeyError(f"{data2_key} not in {model2}.obs")
		if group_key in data1.obs.columns:
			cell_types = data1.obs[group_key]
		elif group_key in data2.obs.columns:
			cell_types = data2.obs[group_key]
		else:
			raise KeyError(f"{group_key} not in {model1}.obs nor in {model2}.obs")
	
		if save:
			if saving_path is None:
				saving_path = os.path.join(os.getcwd(), "results")
			if not os.path.exists(saving_path):
				os.mkdir(sabing_path)

		entropy1 = pd.merge(data1.obs[data1_key], cell_types, how="left", right_index=True, left_index=True)
		entropy1.columns = ["entropy", entropy1.columns[-1]]
		entropy2 = pd.merge(data2.obs[data2_key], cell_types, how="left", right_index=True, left_index=True)
		entropy2.columns = ["entropy", entropy2.columns[-1]]
		entropy1["model"] = [model1] * len(entropy1)
		entropy2["model"] = [model2]*len(entropy2)
		entropies= pd.concat((entropy1, entropy2))

		g = sns.catplot(entropies, x="entropy", y=group_key, hue="model", kind="violin", inner="quart", split=True)
		g.fig.suptitle(f"Entropy distribution")
		if save:
			plt.savefig(os.path.join(saving_path, f"entropies.png"))
		
  				

	def plot_probability_distribution(self, data1:Union[MuData,AnnData], data2:Union[MuData,AnnData], group_key:str,
						 data1_key:str="palantir_fate_probabilities",
						 data2_key:str="palantir_fate_probabilities", 
						 model1: str = "multiomics",
						 model2: str = "rna",
						 save:bool=True, saving_path: Optional[str] = None):
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
		- save: bool
			Whether to save results in saving_path. Default is True.
		- saving_path: str, optional
			Path where results are stored. If not provided then cwd()/results.
		"""
		if data1_key not in data1.obsm.keys():
			raise KeyError(f"{data1_key} not in {model1}.obsm")	
		if data2_key not in data2.obsm.keys():
			raise KeyError(f"{data2_key} not in {model2}.obsm")

		if group_key in data1.obs.columns:
			cell_types = data1.obs[group_key]
		elif group_key in data2.obs.columns:
			cell_types = data2.obs[group_key]
		else:
			raise KeyError(f"{group_key} not found in any data.obs")

		if save:
			if saving_path is None:
				saving_path = os.path.join(os.getcwd(), "results")
			if not os.path.exists(saving_path):
				os.mkdir(saving_path)

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
			if save:
				plt.savefig(os.path.join(saving_path, f"fateprobs_{celltype}.png"))		



class LinearModel:
	def __init__(self, data: pd.DataFrame, group_key:str, save:bool = True, saving_path: Optional[str]=None):
		self.data = data
		self.group_key = group_key
		if save:
			if saving_path is None:
				saving_path = os.path.join(os.getcwd(), "results")
			if not os.path.exists(saving_path):
				os.mkdir(saving_path)
		self.save = save
		self.saving_path = saving_path 
		
	def fit(self, formula: str):
		for celltype in self.data[self.group_key].unique():
			f = self.data[self.data[self.group_key]==celltype] # Subset for a specific celltype
			model = smf.ols(f"{formula}", data=f).fit()
			summary = model.summary()
			if self.save:
				resultDf = pd.DataFrame(summary.tables[1].data)
				resultDf.to_csv(os.path.join(self.saving_path, f"{celltype}.tsv"), sep="\t")		

				

class FatesLinearModel(LinearModel):
	"""
    Fits linear model on fate probability comparing its dependence on model type (e.g., multiomics, rna, etc.) and terminal state for every celltype.
   	Params:
    --------
	- data1: anndata.AnnData or muon.MuData
		First dataset to compare.
	- data2: anndata.AnnData or muon.MuData
 		Second dataset to compare.
	- group_key: str
		Key in either data1.obs or data2.obs containing cell type labels. 
	- key1: str
		Key in data1.obsm where palantir fates probabilities are stored. Default is "palantir_fate_probabilities".
	- key2: str    
		Key in data2.obsm where palantir fates probabilities are stored. Default is "palantir_fate_probabilities".
	- model1: str
		String specyfing which model is data1. By default "multiomics", suggesting data1 contains results from Palantir algorithm based on multiomics data. 
	- model2: str
		String specifying which model is data2. By default "rna", suggesting data2 contains results from Palantir algorithm based on RNA data. 
	- save: bool 
		Whether to save Linear Model results. By default True.
	- saving_path: str, optional
		Path where results are stored. If not provided, then cwd()/results. 
	"""
	def __init__(self, data1: Union[MuData, AnnData], data2: Union[MuData, AnnData], group_key:str,
				key1: str= "palantir_fate_probabilities", key2: str= "palantir_fate_probabilities",
				model1:str = "multiomics", model2:str="rna", save:bool = True, saving_path: Optional[str]=None):
			
		if key1 not in data1.obsm.keys():
			raise KeyError(f"{key1} not in {model1}.obsm")
		if key2 not in data2.obsm.keys():
			raise KeyError(f"{key2} not in {model2}.obsm")
		if group_key in data1.obs.columns:
			cell_types = data1.obs[group_key]
		elif group_key in data2.obs.columns:
			cell_types = data2.obs[group_key]
		else:
			raise KeyError(f"{group_key} not in .obs")
		
		df1 = pd.merge(data1.obsm[key1], cell_types, how="left", right_index=True, left_index=True)
		df2 = pd.merge(data2.obsm[key2], cell_types, how="left", right_index=True, left_index=True)
		df1["model"] = [model1]*len(df1)
		df2["model"] = [model2]*len(df2)
		df = pd.concat((df1,df2))
		terminalStates = [cell_types.loc[ts] for ts in df.columns[:-2]]
		df.columns = terminalStates + list(df.columns[-2:])
		df = pd.melt(df, id_vars=[group_key, "model"], var_name = "terminal", value_name="probability")
			
		super().__init__(df, group_key, save=save, saving_path=saving_path)
		
	def fit(self):
		formula = "probability~ model + terminal + model:terminal"
		super().fit(formula)



class EntropyLinearModel(LinearModel):                                                                                                                     
	"""     
	Fits linear model on entropy comparing its dependence on model type (e.g., multiomics, rna, etc.) for every celltype.    
	Params:                                                                                                                                              
	--------                                                                                                                                             
	- data1: anndata.AnnData or muon.MuData                                                                                                              
		First dataset to compare.                                                                                                                        
	- data2: anndata.AnnData or muon.MuData
		Second dataset to compare.
	- group_key: str            
		Key in either data1.obs or data2.obs containing cell type labels. 
	- key1: str                 
		Key in data1.obs where palantir entropy values are stored. Default is "palantir_entropy".                                       
	- key2: str                                                                                                                                          
		Key in data2.obsm where palantir entropy values are stored. Default is "palantir_entropy".                                       
	- model1: str
		String specyfing which model is data1. By default "multiomics", suggesting data1 contains results from Palantir algorithm based on multiomics data. 
	- model2: str                                                                                                                                        
		String specifying which model is data2. By default "rna", suggesting data2 contains results from Palantir algorithm based on RNA data.           
	- save: bool 
		Whether to save Linear Model results. By default True.                                                                                           
	- saving_path: str, optional
	Path where results are stored. If not provided, then cwd()/results.                                                                             
	"""
	def __init__(self, data1: Union[MuData, AnnData], data2: Union[MuData, AnnData], group_key:str,                                                      
				key1: str= "palantir_entropy", key2: str= "palantir_entropy",
				model1:str = "multiomics", model2:str="rna", save:bool = True, saving_path: Optional[str]=None):                                       
  
		if key1 not in data1.obs.columns:
			raise KeyError(f"{key1} not in {model1}.obs")
		if key2 not in data2.obs.columns:                                                                                                                       
			raise KeyError(f"{key2} not in {model2}.obs")                                                                                            
		if group_key in data1.obs.columns:
			cell_types = data1.obs[group_key]
		elif group_key in data2.obs.columns:
			cell_types = data2.obs[group_key]                                                                                                            
		else:
			raise KeyError(f"{group_key} not in .obs")

		df1 = pd.merge(data1.obs[key1], cell_types, how="left", right_index=True, left_index=True)
		df1.columns = ["entropy", df1.columns[-1]]
		df2 = pd.merge(data2.obs[key2], cell_types, how="left", right_index=True, left_index=True)
		df2.columns = ["entropy", df2.columns[-1]]
		df1["model"] = [model1]*len(df1)
		df2["model"] = [model2]*len(df2)
		df = pd.concat((df1,df2))
		super().__init__(df, group_key, save=save, saving_path=saving_path)

	def fit(self):
		formula = f"entropy~ model"
		super().fit(formula)
 


