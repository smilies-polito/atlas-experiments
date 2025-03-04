######################################
# Script per heatmap correlazioni pseudotime, pseudotime true, entropia dai risultati di scMultiSim
######################################

import os
import muon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(x, y, saving_path, c=None, title=None, xlabel=None, ylabel=None, cmap_label=None, xlim=None, ylim=None):
	plt.figure(figsize=(8, 6))

	if c is not None:
		scatter = plt.scatter(x,y,c=c, cmap="YlOrBr")
		plt.colorbar(scatter)
	else:
		scatter= plt.scatter(x,y)

	if ylim is not None:
		plt.ylim(ylim[0], ylim[1])
	if xlim is not None:
		plt.xlim(xlim[0], xlim[1])

	if title is not None:
		plt.title(title)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)

	plt.savefig(saving_path)
	plt.close()


if __name__=="__main__":
	data = muon.read_h5mu(...)
	saving_folder = ...
	df = pd.read_csv( ... )
	df.set_index("Unnamed: 0", inplace=True)

	x, y = data.obsm["X_umap"][:,0], data.obsm["X_umap"][:,1]
	entropy = df["entropy"]
	kl = df["KL"]
	scatter_plot(x=x, y=y, saving_path = os.path.join(saving_folder, "entropy_umap.png"), c=entropy, title = "Entropy", xlabel = None, ylabel=None, cmap_label="entropy", xlim = None, ylim= None)
	scatter_plot(x=x, y=y, saving_path = os.path.join(saving_folder, "kldiv_umap.png"), c=entropy, title= "KL divergence", xlabel = None, ylabel=None, cmap_label = "KL", xlim= None, ylim= None)

	pseudotime = data.obs["rna:pseudotime"]
	scatter_plot( x= pseudotime, y = entropy, saving_path = os.path.join(saving_folder, "scatter_entropy.png"), c=None, title = "Correlation pseudotime - entropy", xlabel="Pseudotime", ylabel = "Entropy", cmap_label=None, xlim = (0, 1.1), ylim=None)
	scatter_plot( x = pseudotime, y = kl, saving_path = os.path.join(saving_folder, "scatter_kldiv.png"), c=None, title = "Correlation pseudotime - kl", xlabel = "Pseudotime", ylabel = "KL divergence", cmap_label= None, xlim= (0, 1.1), ylim = None)


	

