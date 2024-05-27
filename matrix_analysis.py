import os
import numpy as np
import scipy as sc
import scanpy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 

from csv import DictWriter
from anndata import AnnData
from typing import Optional, Union

class MatrixAnalyser:
    """
    Class implementing analysis for the graph inuced by the transition matrix. 

    Attributes
    ----------
    _matrix: numpy.ndarray/scipy.sparse.csr_matrix.
        transition matrix
    _G: networkx.Graph
        graph associated with transition matrix.
    _cluster_key: str
        string indicating the cluster in adata.obs.
    _adata: anndata.AnnData
        AnnData associated to the transition matrix.
    _k: int
        size of local neighborhood.
    _pc: int, optional
        number of principal components.
    _lsi: int,optional
        number of LSI dimensions.
    _seed: int
        seed for computations (default 52).
    _sink: list
        list containing the sink nodes on the condensation graph.
    _is_stochastic: bool
        boolean indicating whether the transition matrix is stochastic.
    _params:
        dictionary containing multiple parameters for graph analysis will be updated. 

    """
    def __init__(self, matrix: Union[np.ndarray, sc.sparse.csr_matrix], adata: AnnData, cluster_key: Optional[str]= None, **kwargs):
        """
        Parameters
        ----------
        matrix : numpy.ndarray/scipy.sparse.csr_matrix
            transition matrix
        adata : anndata.AnnData
            AnnData associated to the transition matrix     
        clustr_key: str
            str in adata.obs containing cluster labels. 
        **kwargs: 
            mandatory to specify "k" (int) size of the local neighbohood and at least one between "pc" (int) and "lsi" (int),
            the number of principal compoentns and LSI dimensions.
        

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
        self._seed = 52
        self._sink = [] #will add sink components in condensation graph 
        self._is_stochastic = np.allclose(matrix.sum(axis=1), np.ones(matrix.shape[0])) and not np.sum(np.any(matrix<0))
        self._params = {}

        self._params['k'] = self._k
        self._params['pca'] = self._pc
        self._params['lsi'] = self._lsi

        if not self._is_stochastic:
            raise(ValueError, 'The matrix is not stochastic')


    def _topology_analysis(self):
        """ 

        Function that analyses the graph topology requirements: aperiodicity and connectivity 

        Updates
        -------
        ._params: dict
            adds boolean for "aperiodic" and "strongly_connected"

        """
        self._params['aperiodic'] = nx.is_aperiodic(self._G)
        self._params['strongly_connected'] = nx.is_strongly_connected(self._G)



    def _condensation_graph(self, save_composition: bool = False):
        """
        
        Function that analyses of the condensation graph.
        
        Parameters
        -----------
        save_composition: bool
            indicates whether to save cluster composition of the condensation graph

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

        self._params['#sink'] = len(self._sink)   

        # Plot condensation graph
        fig = plt.figure()
        title, path = self._set_title_and_path(f"Condensation Graph")
        fig.suptitle(title, fontsize=11)
        nx.draw(self._H, pos=nx.spring_layout(self._H, seed=self._seed), node_color=colors, labels=labels)
        fig.savefig(path)
    
        # Plot cluster composition for each node in H
        if self._cluster_key is not None and save_composition: 
            mapping = nx.get_node_attributes(H, 'members')
            cluster_memberships = nx.get_node_attributes(self._G, self._cluster_key)

            for node in self._H.nodes:
                original_nodes = mapping[node]
                composition = [cluster_memberships[n] for n in original_nodes]
                lab, cnt = np.unique(composition, return_counts =True)
                diff = set(self._adata.obs[self._cluster_key].values)-set(lab)
                lab = np.hstack((lab, np.array(list(diff))))
                cnt = np.hstack((cnt, np.zeros(len(diff))))
                fig= plt.figure(figsize=(15,8))
                plt.bar(lab, cnt)
                fig.supxlabel('Cluster Composition')
                fig.supylabel('Frequency')
                title, path = self._set_title_and_path(f"Node {node} composition")
                fig.suptitle(title)
                plt.xticks(range(len(lab)), rotation=30)
                fig.savefig(path)


    def _set_title_and_path(self, string: str):
        """

        Create title and path for figures.

        Parameters
        -----------
        string: str
            string used to compose title and path 

        Returns
        -------
        title: str
            string with image title
        path: str
            string with image path 

        """
        title = string + " " + str(self._k) + "K" if self._k is not None else string
        title = title + " " + str(self._pc) + "PC" if self._pc is not None else title
        title = title + " " + str(self._lsi) + "LSI" if self._lsi is not None else title
        
        path = os.path.join(os.getcwd(), 'figures', title.replace(" ", "").lower() + '.png')
        return title, path


    def _find_invariant(self, plot: bool =True): 
        """

        Find invariant probability distribution
        Parameters
        -----------
        plot: bool
        whether to plot and save 30 mostly probable cells. Default True.

        Updates
        -------
        _params: dict
        adds "stationary_distribution_sum1" decribing if there exists single normalised invariant probability distribution 

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
            self._params['stationary_distrbution_sum1'] = np.isclose(np.sum(self._stationary_distribution), 1)

            # 4. Save results with barcodes labels
            df = pd.DataFrame({}, index=nx.get_node_attributes(sg,'barcode').values())
            df['probability'] = self._stationary_distribution

            #Plot highly probable cells on umap for report
            if plot:
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
            self._params['stationary_distrbution_sum1'] = None
    

    def _save_params(self, path: str):
        """ 

        Saves params to csv. The file is created if is doesn't exists, otherwise data is appended to the existing path.
        Parameters
        -----------
        path: str
            path of csv where to store results. 

        """
        if not os.path.exists(path):
            with open(path, 'w') as f:
                writer_object = DictWriter(f, fieldnames=self._params.keys())
                writer_object.writeheader()
                writer_object.writerow(self._params)
                f.close()
        else:
            with open(path, 'a', newline='') as f:
                writer_object = DictWriter(f, fieldnames=self._params.keys())
                writer_object.writerow(self._params)
                f.close()
    


