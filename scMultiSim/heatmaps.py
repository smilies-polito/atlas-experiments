import os
import itertools
import numpy as np
import pandas as pd
from core.utils import compute_branch_correlation


if __name__=="__main__":
		
	seed=42
	np.random.seed(seed)
	state = np.random.get_state()
	
	branch = {"4-5-2": ["4_5", "5_2"], "4-5-3": ["4_5", "5_3"], "4-1": ["4_1"]}
	grid = {"num_waypoints": [500], 
		"knn" : [100, 200, 30]}


	for values in itertools.product(*grid.values()):
		n_waypoints, knn = values
		folder = os.path.join(os.getcwd(), "results_grid", f"{n_waypoints}_{knn}_multiomics")
		results = pd.read_csv(os.path.join(folder, f"results_{n_waypoints}_{knn}.tsv"), sep="\t", header=0, index_col=0)
		simple_correlation = compute_branch_correlation(dataframe=results, method_key = "spearman", key1="rna:pseudotime",key2="palantir_pseudotime", group_key="rna:pop", branch=branch, plot=False)
		simple_correlation = simple_correlation.T
		simple_correlation["knn"] = knn
		simple_correlation["n_waypoints"] = n_waypoints
		print(simple_correlation)
		
		
