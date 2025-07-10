import os
import json
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


def save_run_results(dictionary:dict, dataframe:pd.DataFrame, saving_folder:str, terminal:bool=True):
	path = os.path.join(saving_folder, "results_terminal.json") if terminal else os.path.join(saving_folder, "results_noterminal.json")
	with open(path, "w") as f:
		json.dump(dictionary, f)
		f.close()
	path = os.path.join(saving_folder, "results_terminal.tsv") if terminal else os.path.join(saving_folder, "results_noterminal.tsv")
	dataframe.to_csv(path, sep="\t", header=True, index=True)
	

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
		except Exception as e:
			print("Figure not saved. check path", e)
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
	


	



