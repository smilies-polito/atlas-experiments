import os
import celltypist 
import muon as mu 
import seaborn as sns
import matplotlib.pyplot as plt
from celltypist import models


if __name__=="__main__":
	data = mu.read_h5mu( ... )
	rna = data["rna"]
	predictions = celltypist.annotate(rna, model="Pan_Fetal_Human.pkl", majority_voting=True, mode="best match")
	predictions = predictions.to_adata()
	data.obs = data.obs.merge(predictions.obs, how="left", left_index=True, right_index=True)
	mu.pl.embedding(data, basis="X_umap", color=["STD.CellType", "majority_voting"], save = ... , show=False, legend_loc="on data") 
	celltags = data.obs.groupby(by=["STD.CellType", "majority_voting"]).size().reset_index(name="counts")
	celltags = celltags.pivot_table(values="counts", index="STD.CellType", columns="majority_voting", fill_value=0)
	celltags = (celltags - celltags.min().min())/(celltags.max().max() - celltags.min().min())	
	
	plt.figure(figsize=(13,13))
	sns.heatmap(celltags, annot=False, vmin=0, vmax=1, cbar_kws={"label":"Normalised Counts"})
	plt.xlabel("Celltypist Majority Voting Labels")
	plt.ylabel("Redeem Original Labels")
	plt.title("Heatmap Celltypist Labels vs Original Labels")
	plt.savefig( ... ) 
