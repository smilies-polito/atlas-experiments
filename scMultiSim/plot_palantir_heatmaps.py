import os
import json
import numpy as np
import pandas as pd
from core.utils import simple_heatmap
from itertools import product

def fill_with_results(dictionary, diff_cif_fraction, sigma_cif, values, has_pvalue:bool=True):
	dictionary["diff_cif_fraction"].append(diff_cif_fraction)
	dictionary["sigma_cif"].append(sigma_cif)
	if has_pvalue:
		dictionary["statistic"].append(values[0])
		dictionary["pvalue"].append(values[1])
	else:
		dictionary["statistic"].append(values)

def plot_heatmap(dataframe, has_pvalue, saving_path, **kwargs):
	if has_pvalue:
		data = dataframe[["diff_cif_fraction", "sigma_cif", "statistic"]].pivot(index="diff_cif_fraction", columns="sigma_cif", values = "statistic")
		pvalue = dataframe[["diff_cif_fraction", "sigma_cif", "pvalue"]].pivot(index="diff_cif_fraction", columns="sigma_cif", values = "pvalue")
		mask = (pvalue >= 0.5).values
	else:
		data = dataframe[["diff_cif_fraction", "sigma_cif", "statistic"]].pivot(index="diff_cif_fraction", columns="sigma_cif", values = "statistic")
		mask = np.zeros(data.shape).astype(bool)

	simple_heatmap(data.values, mask, annot=True, save = True, saving_path = saving_path, xticklabels = data.columns, yticklabels = data.index,  **kwargs) 	

if __name__=="__main__":
	problematic_configurations = [ ... ]
	
	grid = {"diff_cif_fraction": [.1,.3,.5,.7,.9], "sigma_cif": [.1, .3, .5, .7, .9]}
	model = ...  

	# correlations
	pearson_entropy_terminal = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	kendall_entropy_terminal = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	pearson_entropy = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	kendall_entropy = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	
	pearson_pseudotime_terminal = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	kendall_pseudotime_terminal = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	pearson_pseudotime = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	kendall_pseudotime = {"diff_cif_fraction":[], "sigma_cif":[], "statistic":[], "pvalue":[]}
	
	# f1 score
	cosine_61 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	cosine_679 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	cosine_6782 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	cosine_6783 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	euclidean_679 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	euclidean_61 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	euclidean_6782 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	euclidean_6783 = {"diff_cif_fraction": [], "sigma_cif":[], "statistic":[]}	
	
	for diff_cif_fraction, sigma_cif in product(*grid.values()):
		if (diff_cif_fraction, sigma_cif) in problematic_configurations:
			continue
	
		path = ... 
		with open(path, "r") as f:
			results = json.load(f)
			f.close()
					
		fill_with_results(pearson_entropy_terminal, diff_cif_fraction, sigma_cif, (results["si_terminal"]["pearson_entropy"]["statistics"], results["si_terminal"]["pearson_entropy"]["pvalue"]), has_pvalue=True)
		fill_with_results(pearson_entropy, diff_cif_fraction, sigma_cif, (results["no_terminal"]["pearson_entropy"]["statistics"], results["no_terminal"]["pearson_entropy"]["pvalue"]), has_pvalue=True)
		fill_with_results(kendall_entropy_terminal, diff_cif_fraction, sigma_cif, (results["si_terminal"]["kendall-tau_entropy"]["statistics"], results["si_terminal"]["kendall-tau_entropy"]["pvalue"]), has_pvalue=True)
		fill_with_results(kendall_entropy, diff_cif_fraction, sigma_cif, (results["no_terminal"]["kendall-tau_entropy"]["statistics"], results["no_terminal"]["kendall-tau_entropy"]["pvalue"]), has_pvalue=True)

		fill_with_results(pearson_pseudotime_terminal, diff_cif_fraction, sigma_cif, (results["si_terminal"]["pearson_pseudotime"]["statistics"], results["si_terminal"]["pearson_pseudotime"]["pvalue"]), has_pvalue=True)
		fill_with_results(pearson_pseudotime, diff_cif_fraction, sigma_cif, (results["no_terminal"]["pearson_pseudotime"]["statistics"], results["no_terminal"]["pearson_pseudotime"]["pvalue"]), has_pvalue=True)
		fill_with_results(kendall_pseudotime_terminal, diff_cif_fraction, sigma_cif, (results["si_terminal"]["kendall-tau_pseudotime"]["statistics"], results["si_terminal"]["kendall-tau_pseudotime"]["pvalue"]), has_pvalue=True)
		fill_with_results(kendall_pseudotime, diff_cif_fraction, sigma_cif, (results["no_terminal"]["kendall-tau_pseudotime"]["statistics"], results["no_terminal"]["kendall-tau_pseudotime"]["pvalue"]), has_pvalue=True)

		fill_with_results(cosine_61, diff_cif_fraction, sigma_cif, results["f1"]["6-1"]["cosine"], has_pvalue=False)
		fill_with_results(cosine_6782, diff_cif_fraction, sigma_cif, results["f1"]["6-7-8-2"]["cosine"], has_pvalue=False)
		fill_with_results(cosine_6783, diff_cif_fraction, sigma_cif, results["f1"]["6-7-8-3"]["cosine"], has_pvalue=False)
		fill_with_results(cosine_679, diff_cif_fraction, sigma_cif, results["f1"]["6-7-9"]["cosine"], has_pvalue=False)

		fill_with_results(euclidean_61, diff_cif_fraction, sigma_cif, results["f1"]["6-1"]["euclidean"], has_pvalue=False)
		fill_with_results(euclidean_6782, diff_cif_fraction, sigma_cif, results["f1"]["6-7-8-2"]["euclidean"], has_pvalue=False)
		fill_with_results(euclidean_6783, diff_cif_fraction, sigma_cif, results["f1"]["6-7-8-3"]["euclidean"], has_pvalue=False)
		fill_with_results(euclidean_679, diff_cif_fraction, sigma_cif, results["f1"]["6-7-9"]["euclidean"], has_pvalue=False)

	# Turn into Dataframes
	cosine_61 = pd.DataFrame(cosine_61)
	cosine_679 = pd.DataFrame(cosine_679)	
	cosine_6782 = pd.DataFrame(cosine_6782)	
	cosine_6783 = pd.DataFrame(cosine_6783)

	euclidean_61 = pd.DataFrame(euclidean_61)
	euclidean_679 = pd.DataFrame(euclidean_679)	
	euclidean_6782 = pd.DataFrame(euclidean_6782)	
	euclidean_6783 = pd.DataFrame(euclidean_6783)

	pearson_entropy_terminal = pd.DataFrame(pearson_entropy_terminal)	
	pearson_entropy = pd.DataFrame(pearson_entropy)	
	kendall_entropy_terminal = pd.DataFrame(kendall_entropy_terminal)	
	kendall_entropy = pd.DataFrame(kendall_entropy)	

	pearson_pseudotime_terminal = pd.DataFrame(pearson_pseudotime_terminal)	
	pearson_pseudotime = pd.DataFrame(pearson_pseudotime)	
	kendall_pseudotime_terminal = pd.DataFrame(kendall_pseudotime_terminal)	
	kendall_pseudotime = pd.DataFrame(kendall_pseudotime)

	# Plot heatmap f1	
	kwargs = {"title": "Cosine Similarity Branch:61", "cmap_label": "f1_score", "cmap_range": (-1, 1), "xlabel":"sigma_cif", "ylabel":"rd"} 
	plot_heatmap(cosine_61, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_cosine_61.png"), **kwargs)
	kwargs["title"] = "Cosine Similarity Branch:679"
	plot_heatmap(cosine_679, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_cosine_679.png"), **kwargs)
	kwargs["title"] = "Cosine Similarity Branch:6783"
	plot_heatmap(cosine_6783, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_cosine_6783.png"), **kwargs)
	kwargs["title"] = "Cosine Similarity Branch:6782"
	plot_heatmap(cosine_6782, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_cosine_6782.png"), **kwargs)

	kwargs = {"title": "Euclidean Similarity Branch:61", "cmap_label": "f1_score", "cmap_range": (-50, 50), "xlabel":"sigma_cif", "ylabel":"rd"} 
	plot_heatmap(euclidean_61, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_euclidean_61.png"), **kwargs)
	kwargs["title"] = "Euclidean Similarity Branch:679"
	plot_heatmap(euclidean_679, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_euclidean_679.png"), **kwargs)
	kwargs["title"] = "Euclidean Similarity Branch:6783"
	plot_heatmap(euclidean_6783, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_euclidean_6783.png"), **kwargs)
	kwargs["title"] = "Euclidean Similarity Branch:6782"
	plot_heatmap(euclidean_6782, has_pvalue=False, saving_path = os.path.join(os.getcwd(), "results", f"{model}_euclidean_6782.png"), **kwargs)


	# Plot heatmap correlations
	kwargs = {"title": "Pearson Correlation True Pseudotime - Entropy", "cmap_label": "correlation", "cmap_range": (-1, 1), "xlabel":"sigma_cif", "ylabel":"rd"} 
	plot_heatmap(pearson_entropy_terminal, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_pearson_entropy_terminal.png"), **kwargs)
	plot_heatmap(pearson_entropy, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_pearson_entropy.png"), **kwargs)

	kwargs = {"title": "Pearson Correlation True Pseudotime - Pseudotime", "cmap_label": "correlation", "cmap_range": (-1, 1), "xlabel":"sigma_cif", "ylabel":"rd"} 
	plot_heatmap(pearson_pseudotime_terminal, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_pearson_pseudotime_terminal.png"), **kwargs)
	plot_heatmap(pearson_pseudotime, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_pearson_pseudotime.png"), **kwargs)


	kwargs = {"title": "Kendall-Tau Correlation True Pseudotime - Pseudotime", "cmap_label": "correlation", "cmap_range": (-1, 1), "xlabel":"sigma_cif", "ylabel":"rd"} 
	plot_heatmap(kendall_pseudotime_terminal, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_kendall_pseudotime_terminal.png"), **kwargs)
	plot_heatmap(kendall_pseudotime, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_kendall_pseudotime.png"), **kwargs)
	
	kwargs = {"title": "Kendall-Tau Correlation True Pseudotime - Entropy", "cmap_label": "correlation", "cmap_range": (-1, 1), "xlabel":"sigma_cif", "ylabel":"rd"} 
	plot_heatmap(kendall_entropy_terminal, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_kendall_entropy_terminal.png"), **kwargs)
	plot_heatmap(kendall_entropy, has_pvalue=True, saving_path = os.path.join(os.getcwd(), "results", f"{model}_kendall_entropy.png"), **kwargs)
