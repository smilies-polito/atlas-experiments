import os
import numpy as np 
import pandas as pd
from muon import MuData
from anndata import AnnData
from scipy.sparse import csr_matrix
from atlas import ATLAS, _assign_state_colors


def add_perpendicular_noise(x, y, scale):
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2)
    nx = -dy / norm
    ny = dx / norm
    noise = np.random.normal(0, scale, len(x))
    return x + noise * nx, y + noise * ny


if __name__=="__main__":
	rng = np.random.default_rng(42)
	n_genes = 30	
	n_trunk = 40
	n_branch = 30  # per ramo
	total_cells = n_trunk + n_branch * 2
	cell_labels = [f"cell{i}" for i in range(total_cells)]
	gene_labels = [f"gene{i}" for i in range(n_genes)]
	noise_scale = 0.07

	
	# umap construction
	t_trunk = np.linspace(0, 1, n_trunk)
	x_trunk = t_trunk * 4
	y_trunk = np.zeros_like(t_trunk)
	x_trunk, y_trunk = add_perpendicular_noise(x_trunk, y_trunk, noise_scale)
	x0, y0 = x_trunk[-1], y_trunk[-1]

	t_a = np.linspace(0, 1, n_branch)
	x_a = x0 + t_a * 3
	y_a = y0 + t_a * 1.5
	x_a, y_a = add_perpendicular_noise(x_a, y_a, noise_scale)	

	t_b = np.linspace(0, 1, n_branch)
	x_b = x0 + t_b * 3
	y_b = y0 - t_b * 1.5
	x_b, y_b = add_perpendicular_noise(x_b, y_b, noise_scale)

	umap = pd.DataFrame({ 
		"UMAP1": np.concatenate([x_trunk, x_a, x_b]),
		"UMAP2": np.concatenate([y_trunk, y_a, y_b]),
	 	"state":	(["branchC"] * n_trunk +
				["branchA"] * n_branch +
				["branchB"] * n_branch )}
			, index=cell_labels)


 	# pseudotime
	pt = np.zeros(total_cells)
	for i, cell in enumerate(umap.index):
		x, y = umap.loc[cell, ["UMAP1", "UMAP2"]]
		state = umap.loc[cell, "state"]
		if state == "branchA" or state=="branchB":
			pt[i] = np.sqrt((x - x0) ** 2 + (y- y0)**2)
		else:
			pt[i] = max(0, x0 - x)

	pseudotime = (pt - pt.min()) / (pt.max() - pt.min())
	entropy = 1-pseudotime

	# fate probabilities
	fateA = np.zeros(total_cells)
	fateB = np.zeros(total_cells)
	fateC = np.zeros(total_cells)
	eps = 1e-6
	
	for i, cell in enumerate(umap.index):
		pt = pseudotime[i]
		state = umap.loc[cell, "state"]

		if state == "branchA":
			fateA[i] = 1/3 + (2/3) * pt
			fateB[i] = 1/3 - (1/3) * pt
			fateC[i] = 1/3 - (1/3) * pt
		if state == "branchB":
			fateA[i] = 1/3 - (1/3) * pt
			fateB[i] = 1/3 + (2/3) * pt
			fateC[i] = 1/3 - (1/3) * pt
		if state == "branchC":
			fateA[i] = 1/3 - (1/3) * pt
			fateB[i] = 1/3 - (1/3) * pt
			fateC[i] = 1/3 + (2/3) * pt
		
	mask_center = pseudotime < 0.1
	fateA[mask_center] += rng.uniform(-eps, eps, mask_center.sum())
	fateB[mask_center] += rng.uniform(-eps, eps, mask_center.sum())
	fateC[mask_center] += rng.uniform(-eps, eps, mask_center.sum())
	total = fateA + fateB + fateC 
	fateA /= total
	fateB /= total
	fateC /= total
	fate_probs = pd.DataFrame({
				"branchA": fateA,
				"branchB": fateB, 
				"branchC": fateC}, 
				index = cell_labels)
	# mudata construction
	rna_matrix = np.zeros(shape=(total_cells, n_genes))
	activity_matrix = np.zeros(shape=(total_cells, n_genes))

	rna = AnnData(X=csr_matrix(rna_matrix), 
						obs=pd.DataFrame(index= cell_labels),
						var = pd.DataFrame(index=gene_labels))
	activity = AnnData(X=csr_matrix(activity_matrix), 
						obs=pd.DataFrame(index= cell_labels),
						var = pd.DataFrame(index=gene_labels))

	data = MuData({"rna":rna, "activity":activity})
	data.obs["cluster"] = umap["state"]
	data.obs["pseudotime"] = pseudotime
	data.obs["kl_divergence"] = entropy
	data.obsm["X_umap"] = umap[["UMAP1", "UMAP2"]].values


	# identification of terminal and initial states
	data.uns["initial_states"] = {"branchC": [cell_labels[np.argmin(pseudotime)], cell_labels[np.argmin(pseudotime)]]}
	terminal_states = {}
	for branch in ["branchA", "branchB", "branchC"]:
		mask = umap["state"].values == branch
		idx = np.argmax(pseudotime[mask])
		terminal_states[branch] = umap.index[mask][idx]

	data.uns["terminal_states"] = terminal_states

	_assign_state_colors(data)
	data.obsm["fate_probabilities"] = fate_probs	


	atlas = ATLAS(mudata = data, method = "palantir", fragment_path = None, random_state = 42)	
	atlas._impl.pseudotime_key = "pseudotime"
	atlas._impl.fate_probability_key = "fate_probabilities"

	atlas.plot_tree(nodes=50, save="scfates.png", color="cluster")






