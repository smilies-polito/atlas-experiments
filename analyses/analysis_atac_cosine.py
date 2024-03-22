import os
import numpy as np
import scipy as sc
import pandas as pd
# import scanpy as sp
import networkx as nx
import matplotlib.pyplot as plt

seed = 52

path = os.path.join(os.getcwd(), 'atac_cosine.npz')
labels_path = os.path .join(os.getcwd(), 'atac_barcodes.csv')
matrix = sc.sparse.load_npz(path)

labels = pd.read_csv(labels_path, header=0)
labels.columns = ['barcode', 'seurat', 'louvain']
clusters = set(labels['louvain'].unique())

labels = labels.T.to_dict('dict')

# Constuct graph and add labels to nodes
G = nx.DiGraph(matrix)

for node in labels.keys():
    G.nodes[node]['barcode'] = labels[node]['barcode']
    G.nodes[node]['seurat'] = labels[node]['seurat']
    G.nodes[node]['louvain'] = labels[node]['louvain']


# Stochastic: non-negative square matrices statsfying P1=1
print("Stochastic Matrix:", np.allclose(matrix.sum(axis=1), np.ones(len(labels))) and not np.sum(np.any(matrix<0)))

# Aperiodicity 
print("Aperiodic:", nx.is_aperiodic(G))

# Irreducibility: matrix P is said to be irreducible if the underlying graph G is strongly connected. 
# A graph G is strongly connected if given any two nodes i and j we have that i is reacheble from j. 
# A graph G is strongly connected if its diameter is finite 
print("G strongly connected:", nx.is_strongly_connected(G))

# Problem: G is not strongly connected, hence the matrix P is not irreducible. We need to study the condensation graph:
# There are as many stationary distributions as sink nodes in the codensation graph. 
H = nx.condensation(G)
mapping = nx.get_node_attributes(H, 'members')
louvain = nx.get_node_attributes(G, 'louvain')

# Set labels and colors according to the type of node in H: source, sink, other
labels={}
colors = [] 
for node in H.nodes:
    if(H.out_degree[node]==0):
        colors.append('gold')
        labels[node] = node
    elif(H.in_degree[node]==0):
        colors.append('crimson')
        labels[node] = node
    else:
        colors.append('powderblue')
        labels[node] = node

# Plot condensation graph
fig = plt.figure()
title = "Condensation Graph for the scATAC-seq data transition matrix"
fig.suptitle(title, fontsize=11)
nx.draw(H, pos=nx.spring_layout(H, seed=seed), node_color=colors, labels=labels)
fig.savefig(os.path.join(os.getcwd(),'figures','condensation_graph.png'))

# Plot louvain composition for each notde in H
for node in H.nodes:
    original_nodes = mapping[node]
    louvain_composition = [louvain[n] for n in original_nodes]
    lab, cnt = np.unique(louvain_composition, return_counts =True)
    diff = clusters-set(lab)
    lab = np.hstack((lab, np.array(list(diff))))
    cnt = np.hstack((cnt, np.zeros(len(diff))))
    fig= plt.figure()
    plt.bar(lab, cnt)
    fig.supxlabel('Louvain cluster')
    fig.supylabel('Frequency')
    fig.suptitle(f'Composition for node {node}')
    plt.xticks(range(len(lab)), lab.astype(np.int8), rotation=45)
    fig.savefig(os.path.join(os.getcwd(), "figures", f"node{node}_frequencies.png"))

print("Number of cells in sink node of condensation graph:", len(mapping[0]))


# Find invariant probability distribution of G
# 1. subset matrix for nodes in sink of HG
sg_matrix = matrix[list(mapping[0]),:][:, list(mapping[0])]
sg = G.subgraph(mapping[0])

# 2. Compute first 6 eigenvalues and eigenvectors
eigenvalues, eigenvectors = sc.sparse.linalg.eigs(sg_matrix.transpose())
 
# 3. Get the eigenvector associatd with eigenvalue 1 + normalize
index = np.where(np.isclose(eigenvalues, 1))
stationary_distribution = eigenvectors[:, index].real.flatten()
stationary_distribution = stationary_distribution/np.sum(stationary_distribution)
print("Check pi sums to 1: ", np.isclose(np.sum(stationary_distribution), 1))

# 4. Save results with barcodes labels
df = pd.DataFrame({}, index=nx.get_node_attributes(sg,'barcode').values())
df['probability'] = stationary_distribution

print("Save to .csv")
df = df.sort_values(by="probability", ascending=False)
df.to_csv("stationary_distribution_atac_cosine.csv", sep=',', header=True, index=True)

# Plot highly probable cells on umap for report
# n_selected = 30
# selected_cells = df.head(n_selected).index.values
# path = os.path.join(os.getcwd(), "adata_folder", "10xMouse_30K10LSI.h5ad")
# atac = sp.read_h5ad(path)
# atac.obs['highly_probable'] = atac.obs_names.isin(selected_cells).astype(pd.Categorical)
# fig, ax = plt.subplots(figsize=(8, 6))
# sp.pl.umap(atac, color='louvain', show=False, ax=ax)
# sp.pl.umap(atac, color=['highly_probable'], title=f"Top {n_selected} highly probable cells in sink component", 
#            show=False, use_raw=False, colorbar_loc = None, ax=ax, cmap="Reds")
# fig.savefig(os.path.join(os.getcwd(), 'figures', f'umaptop{n_selected}highlyprobabile.png'))


