import os
import matplotlib.pyplot as plt 
import seaborn as sns 

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
	ax = sns.heatmap(data, annot=annot, fmt= ".2f",	cmap = cmap, cbar_kws={"label": kwargs.get("cmap_label", "")}, linewidth = .5, linecolor="black", vmin=cmap_range[0], vmax=cmap_range[1])

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
	

	
