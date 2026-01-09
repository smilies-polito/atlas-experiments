import os, json
import muon as mu
import scanpy as sc
import pandas as pd
from unsupervised_metrics import pearson_entropy_pseudotime, spearman_entropy_pseudotime, fate_concentration_index,  terminal_state_silhouette
from palantir_wrapper import PalantirWrapper
from plots import plot_palantir_results
from itertools import product



if __name__=="__main__":

	rd_list = [0.1, 0.3, 0.5, 0.7, 0.9]
	sigma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
	grid = product(rd_list, sigma_list)
	data_folder = os.path.join(os.getcwd(), "data", "phyla3", "phyla3", "30_30_30")
	for rd, sigma in grid:
		data = mu.read_h5mu(os.path.join(data_folder, f"{rd}_{sigma}_data.h5mu"))
		data.obs["developmental stage"] = data.obs["rna:pop"].copy()
		mu.pl.umap(data, color=["developmental stage"], save=f"_{rd}_{sigma}_303030.png")
	exit()
		





	rna_list = [30, 50, 70, 100]
	atac_list = [30, 50, 70, 100]
	wnn_list = [30, 50, 70, 100]
	knn_list = [30, 50, 70, 100]
	nw_list = [500, 750]
	grid = product(rna_list, atac_list, wnn_list, knn_list, nw_list)

	cell_path = os.path.join(os.getcwd(), "data", "phyla3", "selected_cells_palantir.json")		
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	early_cell = list(cells["initial"].values())[0]
	results_path = os.path.join(os.getcwd(), "output", "phyla3", "03_03_results.tsv")

	for rna, atac, wnn, knn, n_waypoints in grid:
		data_folder = os.path.join(os.getcwd(), "data", "phyla3", "phyla3", f"{rna}_{atac}_{wnn}")
		data = mu.read_h5mu(os.path.join(data_folder, "0.3_0.3_data.h5mu"))
		output_folder = os.path.join(os.getcwd(), "output", "phyla3", f"{rna}_{atac}_{wnn}_03_03_{n_waypoints}_{knn}")
		if not os.path.exists(output_folder):
			os.mkdir(output_folder)
		results = {"rna": rna, "atac":atac, "wnn":wnn, "knn_waypoints":knn, "n_waypoints":n_waypoints, "sigma":0.3, "rd":0.3,
				"n_terminal":None,
				"pearson_corr": None, "pearson_pval": None, "pearson_ci": None, 
				"spearman_corr": None, "spearman_pval": None, "spearman_ci": None,
				"concentration_corr": None, "concentration_pval": None, "concentration_ci": None,
				"silhouette_confidence": None, "silhouette_confidence_ncells": None,
				"silhouette_pseudotime": None, 
				"silhouette_soft": None, 
				"local_score": None,
				"failed": False}
				

		mu.pl.umap(data, color=["rna:pop", "rna:pseudotime"], save=f"_{rna}_{atac}_{wnn}_03_03.png")
#		try:
		pw = PalantirWrapper()
		pw.compute_kernel(data)
		pw.run_diffusion_maps(data, seed=42)	
		pw.determine_multiscale_space(data)
		pw.run_palantir(data, early_cell=early_cell, num_waypoints=n_waypoints, knn=knn, seed=42)
		plot_palantir_results(data, modality_key=None, saving_path = output_folder)
#		except:
		results["failed"] = True
		pd.DataFrame(results, index=[0]).to_csv(results_path, sep="\t", header=False, index=False, mode="a")
#		continue
			
		results["n_terminal"] = data.obsm["palantir_fate_probabilities"].shape[1]
		if results["n_terminal"] < 2:
			pd.DataFrame(results, index=[0]).to_csv(results_path, sep="\t", header=False, index=False, mode="a")
			continue

		pearson, ppval, pci = pearson_entropy_pseudotime(entropy=data.obs["palantir_entropy"], pseudotime=data.obs["palantir_pseudotime"])
		results["pearson_corr"] = pearson 
		results["pearson_pval"] = ppval
		results["pearson_ci"] = pci 

		spearman, spval, sci = spearman_entropy_pseudotime(entropy=data.obs["palantir_entropy"], pseudotime=data.obs["palantir_pseudotime"])
		results["spearman_corr"] = spearman  
		results["spearman_pval"] = spval
		results[ "spearman_ci"] = sci 

		correlation, pval, ci, concentration_index = fate_concentration_index(fates=data.obsm["palantir_fate_probabilities"], pseudotime=data.obs["palantir_pseudotime"])
		results["concentration_corr"] = correlation 
		results["concentration_pval"] = pval
		results[ "concentration_ci"] = ci

		data.obs["fate_concentration"] = concentration_index
		mu.pl.umap(data, color="fate_concentration", save="_{rna}_{atac}_{wnn}_03_03_{n_waypoints}_{knn}_fate_concentration.png")

		S, n_committed = terminal_state_silhouette(fates = data.obsm["palantir_fate_probabilities"], soft_assignment=False, pseudotime = data.obs["palantir_pseudotime"], confidence_filtering=True, confidence_parameter=0.7)	
		results["silhouette_confidence"]=S
		results["silhouette_confidence_ncells"]=n_committed
		S, _ = terminal_state_silhouette(fates = data.obsm["palantir_fate_probabilities"], pseudotime_weight=True, pseudotime = data.obs["palantir_pseudotime"])
		results["silhouette_pseudotime"]=S
		S, _ = terminal_state_silhouette(fates = data.obsm["palantir_fate_probabilities"], soft_assignment=True, pseudotime = data.obs["palantir_pseudotime"])
		results["silhouette_soft"]=S

		D = data.obsp["wnn_distances"]
		S = local_fate_biology_coupling(fates = data.obsm["palantir_fate_probabilities"], wnn=D)
		results["local_score"] = S
		

		pd.DataFrame(results, index=[0]).to_csv(results_path, sep="\t", header=False, index=False, mode="a")



