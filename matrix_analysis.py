import os
import numpy as np
import scipy as sc
import scanpy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 

from anndata import AnnData
from typing import Optional, Union

class MatrixAnalyser:
    def __init__(self, matrix: Union[np.ndarray, sc.sparse.csr_matrix], adata: AnnData, cluster_key: Optional[str]= None, **kwargs):
        """
            Construct graph induced by the transitioon matrix + analysis.
            params:
                - matrix: transition matrix.
                - adata: annotated dataset.
                - cluster_key: key in adata.obs indicating the cluster. Default None.
                - kwargs: any parameter such as nmber of lsi, resolution, pcs, k used to configure titles and paths.
        """
        self._matrix = matrix
        self._G = nx.DiGraph(matrix)

        if cluster_key is not None and cluster_key not in adata.obs.columns:
            print("Invalid cluster_key")
            cluster_key = None

        self._cluster_key = cluster_key

        for node in self._G.nodes:
            self._G.nodes[node]['barcode'] = adata.obs.index[node]
            self._G.nodes[node][cluster_key] = adata.obs[cluster_key].iloc[node] if cluster_key is not None else None

        self._adata = adata
        self._k = kwargs['k'] if 'k' in kwargs.keys() else None
        self._lsi = kwargs['lsi'] if 'lsi' in kwargs.keys() else None
        self._pc = kwargs['pc'] if 'pc' in kwargs.keys() else None
        self._res = kwargs['res'] if 'res' in kwargs.keys() else None
        self._seed = 52
        self._sink = [] #will add sink components in condensation graph 
        self._is_stochastic = np.allclose(matrix.sum(axis=1), np.ones(matrix.shape[0])) and not np.sum(np.any(matrix<0))

        if not self._is_stochastic:
            raise(ValueError, 'The matrix is not sotchastic')

    def _topology_analysis(self):
        """ 
        Analysis for Markov Chain. 
        """
        self._aperiodic = nx.is_aperiodic(self._G)
        self._strongly_connected = nx.is_strongly_connected(self._G)

    def _condensation_graph(self, save_composition: bool = False):
        """
            Analysis of the condensation graph.
            params: 
                - save_composition: boolean indicating whether so save cluster composition of the condensation graph
        """
        H = nx.condensation(self._G)
        self._H  = H

        # Set labels and colors according to the type of node in H: source, sink, other
        labels={}
        colors = [] 
        for node in H.nodes:
            if(H.out_degree[node]==0):
                colors.append('gold')
                labels[node] = node
                self._sink.append(node)
            elif(H.in_degree[node]==0):
                colors.append('crimson')
                labels[node] = node
            else:
                colors.append('powderblue')
                labels[node] = node    

        # Plot condensation graph
        fig = plt.figure()
        title, path = self._set_title_and_path("Condensation Graph")
        fig.suptitle(title, fontsize=11)
        nx.draw(self._H, pos=nx.spring_layout(self._H, seed=self._seed), node_color=colors, labels=labels)
        fig.savefig(path)

        mapping = nx.get_node_attributes(H, 'members')
    
        # Plot louvain composition for each node in H
        if self._cluster_key and save_composition: 
            cluster_memberships = nx.get_node_attributes(self._G, self._cluster_key)

            for node in self._H.nodes:
                original_nodes = mapping[node]
                composition = [cluster_memberships[n] for n in original_nodes]
                lab, cnt = np.unique(composition, return_counts =True)
                diff = set(self._adata.obs[self._cluster_key].values)-set(lab)
                lab = np.hstack((lab, np.array(list(diff))))
                cnt = np.hstack((cnt, np.zeros(len(diff))))
                fig= plt.figure()
                plt.bar(lab, cnt)
                fig.supxlabel('Cluster Composition')
                fig.supylabel('Frequency')
                title, path = self._set_title_and_path(f"Node {node} composition")
                fig.suptitle(title)
                plt.xticks(range(len(lab)), lab.astype(np.int8), rotation=45)
                fig.savefig(path)


    def _set_title_and_path(self, string: str):
        """
            Create title and path for figures.
            params:
                - string: string to compose title and path 
        """
        title = string + " " + str(self._k) + "K" if self._k is not None else string
        title = title + " " + str(self._pc) + "PC" if self._pc is not None else title
        title = title + " " + str(self._lsi) + "LSI" if self._lsi is not None else title
        title = title + " " + str(self._res) + "res" if self._res is not None else title
        
        path = os.path.join(os.getcwd(), 'figures', title.replace(" ", "").lower() + '.png')
        return title, path


    def _find_invariant(self): 
        """
            Find invariant probability distribution
        """
        if len(self._sink) == 1:
            # Single probability distribution
            mapping = nx.get_node_attributes(self._H, 'members')[self._sink[0]]
            sg = self._G.subgraph(mapping)
            sg_matrix = self._matrix[list(mapping),:][:, list(mapping)]

            # 2. Compute first 6 eigenvalues and eigenvectors
            eigenvalues, eigenvectors = sc.sparse.linalg.eigs(sg_matrix.transpose())
        
            # 3. Get the eigenvector associatd with eigenvalue 1 + normalize
            index = np.where(np.isclose(eigenvalues, 1))
            stationary_distribution = eigenvectors[:, index].real.flatten()
            self._stationary_distribution = stationary_distribution/np.sum(stationary_distribution)
            print("Check pi sums to 1: ", np.isclose(np.sum(self._stationary_distribution), 1))

            # 4. Save results with barcodes labels
            df = pd.DataFrame({}, index=nx.get_node_attributes(sg,'barcode').values())
            df['probability'] = self._stationary_distribution

            #Plot highly probable cells on umap for report
            n_selected = 30
            selected_cells = df.head(n_selected).index.values
            self._adata.obs['highly_probable'] = self._adata.obs_names.isin(selected_cells).astype(pd.Categorical)
            fig, ax = plt.subplots(figsize=(8, 6))
            sp.pl.umap(self._adata, color=self._cluster_key, show=False, ax=ax)
            title, path = self._set_title_and_path(f"Top {n_selected} cells")
            sp.pl.umap(self._adata, color=['highly_probable'], title=title,  
                show=False, use_raw=False, colorbar_loc = None, ax=ax, cmap="Reds")
            fig.savefig(path)

        else: 
            print("There are multiple sink components in H, not a single invariant probability distribution.")
