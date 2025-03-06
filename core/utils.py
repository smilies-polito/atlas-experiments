import os
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def simple_scatter(x,y,c=None, save:bool=True, saving_path:str=None, **kwargs):

	plt.figure(figsize=(8,6))
	scatter = plt.scatter(x=x, y=y, c=c)

	if c is not None:
		clabel = kwargs.get("cbar_label", "")
		plt.colorbar(scatter, label=clabel)

	if "title" in kwargs:
		plt.title(kwargs["title"])
	if "xlabel" in kwargs:
		plt.xlabel(kwargs["xlabel"])
	if "ylabel" in kwargs:
		plt.ylabel(kwargs["ylabel"])
	
	if "xticks" in kwargs:
		plt.xticks(kwargs["xticks"])
	if "yticks" in kwargs:
		plt.yticks(kwargs["yticks"])
	
	if "xlim" in kwargs:
		plt.xlim(kwargs["xlim"])
	if "ylim" in kwargs:
		plt.ylim(kwargs["ylim"])
		
	if save and saving_path is not None:
		try:
			plt.savefig(saving_path)
		except:
			print("Figure not saved. check path")



def simple_heatmap(data, mask, annot:bool=True, save:bool=True, saving_path:str=None, cmap_range:tuple=(-1,1), **kwargs):
	plt.figure(figsize=(8,6))
	cmap = kwargs.get("cmap", "viridis")
	xticklabels = kwargs.get("xticklabels", "auto")
	yticklabels = kwargs.get("yticklabels", "auto")
	ax = sns.heatmap(data, annot=annot, fmt= ".2f",	cmap = cmap, cbar_kws={"label": kwargs.get("cmap_label", "")}, linewidth = .5, linecolor="black", vmin=cmap_range[0], vmax=cmap_range[1], xticklabels=xticklabels, yticklabels=yticklabels)

	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if mask[i,j]:
				ax.add_patch(plt.Rectangle((j,i), 1,1, fill=False, edgecolor="red", lw=1))
	
	if "title" in kwargs:
		plt.title(kwargs["title"])
	if "xlabel" in kwargs:
		plt.xlabel(kwargs["xlabel"])
	if "ylabel" in kwargs:
		plt.ylabel(kwargs["ylabel"])
	
	if save and saving_path is not None:
		try:
			plt.savefig(saving_path)
		except:
			print("Figure not saved. Check saving path")
	

def compute_branch_correlation(dataframe: pd.DataFrame, method_key:str, key1: str, key2: str, group_key:str, branch:dict, plot:bool= True, save:bool = True, saving_path:str=None, **kwargs): 	
	
	if key1 not in dataframe.columns:
		raise KeyError(f"{key1} not in dataframe.columns")
	if key2 not in dataframe.columns:
		raise KeyError(f"{key2} not in dataframe.columns")	
	if group_key not in dataframe.columns:
		raise KeyError(f"{group_key} not in dataframe.columns")
	
	correlations = {}
	method = pearsonr if method_key=="pearson" else spearmanr
	for k, v in branch.items():
		subsetdata = dataframe[dataframe[group_key].isin(v)]
		correlation = method(subsetdata[key1], subsetdata[key2])
		correlations[k] = (correlation.statistic, correlation.pvalue)	
		
	correlations = pd.DataFrame(correlations, index=["statistic", "pvalue"])
	if plot:
		data = correlations.loc["statistic",:].values.reshape(-1,1)
		mask = (correlations.loc["pvalue",:] >= 0.05).values.reshape(-1,1)
		simple_heatmap(data = data, mask=mask, annot=True, save = save, saving_path= saving_path, cmap_range=(-1,1), **kwargs)
	
	return correlations

def plot_branch_correlation(dataframe: pd.DataFrame, key1:str, key2:str, group_key:str, branch:dict, save:bool=True, saving_path:str = None, **kwargs):
	if key1 not in dataframe.columns:
		raise KeyError(f"{key1} not in dataframe.columns")
	if key2 not in dataframe.columns:
		raise KeyError(f"{key2} not in dataframe.columns")
	if group_key not in dataframe.columns:
		raise KeyError(f"{group_key} not in dataframe.columns")
	title = kwargs.get("title", "")

	for k, v in branch.items():
		subsetdata = dataframe[dataframe[group_key].isin(v)]
		path = os.path.join(saving_path, f"{k}_branch_scatter.png") if saving_path is not None else None
		kwargs["title"] = title + f" {k}" 
		simple_scatter(x=dataframe[key1], y=dataframe[key2], c=None, save=save, saving_path=path, **kwargs)
		
