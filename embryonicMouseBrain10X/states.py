####################################
# Choosing initial and terminal states for Palantirn algorithm
####################################


import os
import json
import scanpy as sc
import numpy as np
import muon as mu 


np.random.seed(52)

data_path = ... 
data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))

ul = np.random.choice(data.obs[data.obs["rna:celltype"]=="Upper Layer"].index.values, 1)
dl = np.random.choice(data.obs[data.obs["rna:celltype"]=="Deeper Layer"].index.values, 1)

astro_genes = [gene for gene in ["Vim", "Gfap", "Aldhl1"] if gene in data.var_names]
opc_genes = [gene for gene in ["Pdgfra", "Olig2", "Sox10"] if gene in data.var_names]
genes = astro_genes + opc_genes
threshold = 0.8
gex_mask = data["rna"][:, genes].X.A > threshold
as_op_cell = np.random.choice(data[gex_mask.all(axis=1)].obs_names, 1)

starting_cell= np.random.choice(data.obs[data.obs["rna:celltype"]=="RG, Astro, OPC"].index.values, 1)[0]
print(starting_cell)
terminal_cells = np.concatenate((ul, dl, as_op_cell), axis=0)

result = {"starting_cell": starting_cell, "terminal_cells": list(terminal_cells)}

data.obs["is_terminal"] = data.obs.index.isin(terminal_cells)
data.obs["is_start"] = data.obs.index==starting_cell
sc.pl.embedding(data, basis="X_umap", color =["is_start","is_terminal"], palette="PuRd", title=["Starting cell", "Terminal cells"])

with open(os.path.join(data_path, "cells.json"), "w") as f:
	json.dump(result, f)
	f.close()
