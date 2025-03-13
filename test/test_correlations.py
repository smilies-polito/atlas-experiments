import os
import numpy as np
import pandas as pd
from core.utils import compute_branch_correlation, plot_branch_correlation, simple_scatter

if __name__=="__main__":
	pseudotime = np.random.uniform(0,1,120)
	entropy = 1 + pseudotime/2
	groups = np.random.choice(["A", "B", "C"], 120)
	colors = pd.Categorical(np.random.choice(["c1", "c2"], 120))
	dataframe = pd.DataFrame({"entropy":entropy, "pseudotime":pseudotime, "groups":groups, "color":colors})
	branch = {"branch1": ["A", "B"], "branch2": ["C"]}

	save = True
	saving_folder = os.path.join(os.getcwd(), "correlations")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)
	
	print(dataframe.head(10), dataframe.shape)
	# TEST HEATMAP CORRELATIONS
	# compute_branch_correlation(dataframe=dataframe, method_key="pearson", key1="pseudotime", key2="entropy", group_key="groups", branch=branch, plot=True, save=True, saving_path=os.path.join(saving_folder, "heatmap_correlations.png"), title="HeatMap Pseudotime - Entropy", xticklabels=[], yticklabels= list(branch.keys()), ylabel = "branch", cmap_label="correlation")   
	print(entropy.max())
	plot_branch_correlation(dataframe=dataframe, key1= "pseudotime", key2= "entropy", group_key = "groups", color_key="color", categorical=True, branch=branch, save=save, saving_path = saving_folder, title= "Pseudotime-Entropy scatter", xlabel = "pseudotime", ylabel="entropy", xlim = (0, 1.1), ylim = (0, dataframe["entropy"].max() + 0.01), legend_loc="lower center", legend_ncols=2) 
	
	simple_scatter(x=dataframe["pseudotime"], y=dataframe["entropy"], c=dataframe["entropy"], saving_path=os.path.join(saving_folder, "scatter.png"))	
