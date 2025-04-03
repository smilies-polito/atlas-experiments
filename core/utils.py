import os
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
from typing import Literal
from skelarn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr, kendalltau

def simple_scatter(x,y, c=None, categorical: bool= False, save:bool=True, saving_path:str=None, **kwargs):

	plt.figure(figsize=(8,6))

	if c is not None:
		if categorical:
			scatter = plt.scatter(x=x, y=y, c=c.cat.codes, cmap="tab20", s=10)
			labels = c.cat.categories
			legend_loc = kwargs.get("legend_loc", "best")
			legend_ncols = kwargs.get("legend_ncols", 1)
			handles, _ = scatter.legend_elements()
			plt.legend(handles, labels, loc=legend_loc, ncols = legend_ncols)
		else:
			scatter = plt.scatter(x=x, y=y, c=c, cmap="YlOrBr", s=10)
			clabel = kwargs.get("cbar_label", "")
			plt.colorbar(scatter, label=clabel)
	else:
		scatter = plt.scatter(x=x, y=y, s=10)

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
	plt.close()



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

	plt.close()
	

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


def compute_correlation(group, method_key: Literal["pearson", "kendall-tau", "spearman"], key1:str, key2:str, return_pvalue: bool = True):
	if method_key == "pearson":
		method = pearsonr
	elif method_key == "kendall-tau":
		method = kendalltau 
	else:
		method = spearmanr

	corr,pvalue = method(group[key1], group[key2])
	if return_pvalue:
		return pvalue
	else:
		return corr 


def compute_f1(fates: pd.DataFrame, y_true:pd.Series):
	y_pred = compute_fates_membership(fates)
	if y_pred is None:
		return -1

	return f1_score(y_true, y_pred, average="macro", labels=y_true.unique())


def compute_fates_membership(fates:pd.DataFrame):
	terminal_states = fates.columns

	if len(terminal_states) == 0: #no terminal states
		return None

	# caso in cui ho probabilità tutte praticamente attaccate allo 0 per tutte le cellule? esistono?
	return fates.idxmax(axis=1) #ritorna primo by default 	
	



