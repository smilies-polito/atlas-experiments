######################################
# Script per heatmap correlazioni pseudotime, pseudotime true, entropia dai risultati di scMultiSim
######################################

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_plot(data, mask, title, xlabel, ylabel, cmap_label, saving_path):
	plt.figure(figsize=(8, 6))
	ax = sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': cmap_label},  mask=mask, linewidths=.5, linecolor='black')

	if mask is not None:
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				if mask[i,j]:
					ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(saving_path)
	plt.close()


if __name__=="__main__":
	data_path = ... 
	df = pd.read_csv(..., sep="\t")
#	df.columns = ["model", "r2", "pearson_statistics", "pearson_pvalue"]	
	df.columns = ["model", "pearson_statistics", "pearson_pvalue"]

	df["num_waypoints"]=df["model"].map(lambda s: s.split("_")[1]).astype(int)
	df["knn"]=df["model"].map(lambda s: s.split("_")[2]).astype(int)
	df["model"]=df["model"].map(lambda s: s.split("_")[0])	

	models = df["model"].unique()
	for model in models:
		correlation = df[df["model"]==model].pivot(index="num_waypoints", columns="knn", values="pearson_statistics")
		pvalues = df[df["model"]==model].pivot(index="num_waypoints", columns="knn", values="pearson_pvalue")
#		r2 = df[df["model"]==model].pivot(index="num_waypoints", columns="knn", values="r2")
		
		heatmap_plot(correlation, (pvalues>=0.5).values, f"Pearson Correlation {model.upper()}", "KNN", "Number of waypoints", "pearson_corr",
				os.path.join(data_path, f"pearson_corr_{model.upper()}.png"))

#		heatmap_plot(r2, None, f"R2 {model.upper()}", "KNN", "Number of waypoints", "r2", os.path.join(data_path, f"r2_{model.upper()}.png"))
