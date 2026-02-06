import os
import numpy as np 
import pandas as pd
import scvelo as scv
import matplotlib.pyplot as plt
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
	gene = "gene1"
	ptf = "gene0"

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
	 	"state":	(["trunk"] * n_trunk +
				["branchA"] * n_branch +
				["branchB"] * n_branch )}
			, index=cell_labels)

	pt = umap["UMAP1"]
	pseudotime = (pt - pt.min()) / (pt.max() - pt.min())
	entropy = 1-pseudotime

	# fate probabilities
	fateA = np.zeros(total_cells)
	fateB = np.zeros(total_cells)
	for i, cell in enumerate(umap.index):
		pt = pseudotime.loc[cell]
		state = umap.loc[cell, "state"]
		if state == "trunk":
			eps = 1e-6
			fateA[i] = 0.5 + rng.uniform(-eps, eps)
			fateB[i] = 0.5 + rng.uniform(-eps, eps) 
		elif state == "branchA":
			fateA[i] = 0.5 + 0.5*pt
			fateB[i] = 0.5 - 0.5*pt
		elif state == "branchB":
			fateA[i] = 0.5 - 0.5*pt 
			fateB[i] = 0.5 + 0.5*pt

	total = fateA + fateB
	fateA /= total
	fateB /= total
	fate_probs = pd.DataFrame({"branchA": fateA, "branchB": fateB}, index = cell_labels)

	# biological signals 
	gexA = 2*pseudotime + 0.3 * np.sin(2* np.pi * pseudotime)
	gexB = 2*pseudotime - 0.3 * np.sin(2* np.pi * pseudotime)
	gex = (fate_probs["branchA"].values * gexA + 
				fate_probs["branchB"].values * gexB)
	gex += rng.normal(0, 0.1, size=total_cells)
		
	actA = 1/(1 + np.exp(-10* (pseudotime-0.35)))
	actB = 1/(1 + np.exp(-10* (pseudotime-0.65)))
	act = (fate_probs["branchA"].values * actA + 
				fate_probs["branchB"].values * actB)
	act+= rng.normal(0, 0.05, size = total_cells)

	
	# MuData construction
	rna_matrix = np.zeros(shape=(total_cells, n_genes))
	activity_matrix = np.zeros(shape=(total_cells, n_genes))
	ptf_idx = gene_labels.index(ptf)
	gene_idx = gene_labels.index(gene)
	rna_matrix[:, ptf_idx] = gex
	activity_matrix[:, gene_idx] = act

	rna = AnnData(X=csr_matrix(rna_matrix), 
						obs=pd.DataFrame(index= cell_labels),
						var = pd.DataFrame(index=gene_labels))
	activity = AnnData(X=csr_matrix(activity_matrix), 
						obs=pd.DataFrame(index= cell_labels),
						var = pd.DataFrame(index=gene_labels))
	data = MuData({"rna":rna, "activity":activity})
	data.obs["cluster"] = umap["state"]
	data.obs["pseudotime"] = pseudotime
	data.obs["entropy"] = entropy
	data.obsm["X_umap"] = umap[["UMAP1", "UMAP2"]].values

	# identification of terminal and initial state probabilities
	data.uns["initial_states"] = {"trunk": umap["UMAP1"].idxmin()}
	data.uns["terminal_states"] = {"branchA": umap.loc[umap["state"] == "branchA", "UMAP1"].idxmax(), 
									"branchB": umap.loc[umap["state"] == "branchB", "UMAP2"].idxmax()} 
	_assign_state_colors(data)
	data.obsm["fate_probabilities"] = fate_probs	


	# initialize ATLAS object 
	atlas = ATLAS(mudata = data, method = "palantir", fragment_path = None, random_state = 42)	
	atlas._impl.pseudotime_key = "pseudotime"
	atlas._impl.fate_probability_key = "fate_probabilities"
#	saving = "pseudotime.png"
#	atlas.plot_embedding(embedding_key = "X_umap", observation= "pseudotime", save = saving)
#	atlas.plot_embedding(embedding_key = "X_umap", observation= "entropy")
#	saving = "fate_probabilities.png"
#	atlas.plot_fate_probabilities(embedding_key = "X_umap", states= None, save = saving)
	saving = f"{ptf}_{gene}"
	atlas.plot_trends(ptf=ptf, gene=gene, save=saving)

		
	
