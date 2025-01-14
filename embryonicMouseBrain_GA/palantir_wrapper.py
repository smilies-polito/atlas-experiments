###########################################################################
# Palantir Wrapper class to call Palantir routines using muon.MuData object
###########################################################################

import os
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import muon as mu
import palantir
from typing import Union, Optional
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix, find


class PalantirWrapper():
	def compute_kernel(self, data: MuData,	
						knn_key: str = "wnn",	
						distance_key: str="wnn_distances",  
						knn:Optional[int]=None, 
						alpha:float=0,
						kernel_key : str="DM_Kernel"):
		"""
		Adapted computation of the gaussian kernel that allows for muon.MuData obejct to be used and wnn. 
		Follows Palantir implementation.
		Params:
		--------
		data: muon.MuData 
		knn_key: str
			String in data.uns where WNN parameters are set. By default "wnn".
		distance_key: str
			String in data.obsp where knn is stored. By default "wnn_distances". 
		knn: int
			Number of Nearest Neighbors. Default is None. 
		alpha: float 
			Normalization paramenter for the diffusion operator. By default 0.
		kernel_key: str
			Key to store the kernel in the obsp of the data. Default is "DM_Kernel".  
		"""
		if distance_key not in data.obsp.keys(): 
			raise KeyError(f"{distance_key} not in data.obsp")
		if knn is None:
			warnings.warn(f"knn parameter not specified. Looking for data.uns[{knn_key}][""params""][""n_neighbors""]")
			if knn_key not in data.uns.keys():
				raise KeyError(f"{knn_key} not in data.uns")	
			else:
				knn = int(data.uns[knn_key]["params"]["n_neighbors"])


		N = data.shape[0]
		kNN = data.obsp[distance_key]
		# By default Palantir computes the scaling factor as \sigma_i = distance to l-th neighbor, with l<k 
		# By default Palantir computes KNN neighbor from scratch using PCs the knn:int=30 parameter and n_pcs=0 (sc.pp.neighbors) and then sets l=knn/3
		# Here we expect either knn to be set, or to be searched in data.uns[knn_key]["params"]["n_neighbors"] as per muon.pp.neighbors construction.  
		adaptive_k = int(np.floor(knn/3)) #default by Palantir.  
		adaptive_std = np.zeros(N)
		for i in np.arange(N):
			adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i+1]])[adaptive_k - 1]
	
		x,y,dists = find(kNN) # x are row indices, y are column indices and dists the values of non zero entries in csr_matrix
		dists /= adaptive_std[x]
		W = csr_matrix((np.exp(-dists), (x,y)), shape=[N,N])
		kernel = W + W.T
		if alpha > 0:
			D = np.ravel(kernel.sum(axis=1))
			D[D!=0] = D[D!=0]**(-alpha)
			mat = csr_matrix((D, (range(N). range(N))), shape=[N,N])
			kernel = mat.dot(kernel).dot(mat)

		data.obsp[kernel_key] = kernel

	def run_diffusion_maps(self, data: MuData,
							n_components:int=10,
							seed: int = 52, 
							kernel_key: str = "DM_Kernel", 
							sim_key: str= "DM_Similarity",
							eigval_key: str = "DM_EigenValues",
							eigvec_key: str = "DM_EigenVectors"):

		"""
		Wrapper for palantir.utils.diffusion_maps_from_kernel. 
		Params:
		------
		data: muon.MuData
		n_components: int 
			Number of Diffusion components to compute. By default 10.
		seed: int
			Numpy random seed. By deafult 0. 
		kernel_key: str
			Key in data.obsp containing the kernel. By default "DM_Kernel". 
		sim_key: str
			Key in data.obsp where the diffusion operator T is stored. By default "DM_Similarity".
		eigval_key: str
			Key in data.uns where to store the eigevalues. By default "DM_EigenValues".
		eigvec_key: str
			Key in data.obsm where to store the eigenvectors. By default "DM_EigenVectors". 
		"""
		if kernel_key not in data.obsp.keys():
			raise KeyError(f"{kernel_key} not in data.obsp")

		kernel= data.obsp[kernel_key]
		print("Calling Palantir diffusion_maps_from_kernel...")
		res = palantir.utils.diffusion_maps_from_kernel(data.obsp[kernel_key], n_components, seed)
		data.obsp[sim_key] = res["T"] 
		data.obsm[eigvec_key] = res["EigenVectors"].values
		data.uns[eigval_key] = res["EigenValues"].values


	def determine_multiscale_space(self, dm_res: MuData,
									n_eigs: Union[int, None]= None,
									eigval_key: str= "DM_EigenValues",
									eigvec_key: str= "DM_EigenVectors",
									out_key: str = "DM_EigenVectors_multiscaled"):
		"""
		Wrapper for palantir.utils.determine_multiscale_space. 
		Params:
		------
		dm_res: mudata.MuData
		n_eigs: int, optional
			Number of eigen vectors to use. If None the eigen gap heuristic is used. Default is None.
		eigval_key: str
			Key in data.uns where eigenvalues are stored. Default is "DM_EigenValues".
		eigvec_key: str
			Key in data.obsm where eigenvectors are stored. Default is "DM_EigenVectors":
		out_key: str
			Key in data.obsm where the results are stored. Default is "DM_EigenVectors_multiscaled".
		"""
		# Palantir constructs a multi-scale distance to construct a more reliable nearest neighbor graph G_E.
		# G_E is later used for pseudotime computations using waypoints. 
		if eigval_key not in data.uns.keys():
			raise KeyError(f"{eigval_key} not in data.uns")
		if eigvec_key not in data.obsm.keys():
			raise KeyError(f"{eigvec_key} not in data.obsm")
		
		eigenvectors = pd.DataFrame(data.obsm[eigvec_key], index=data.obs_names) if not isinstance(data.obsm[eigvec_key], pd.DataFrame) else data.obsm[eigvec_key] # for compatibility with Palantir framework
		dm_dict = {"EigenValues": data.uns[eigval_key], "EigenVectors": eigenvectors}
		result = palantir.utils.determine_multiscale_space(dm_res = dm_dict, n_eigs=n_eigs, eigval_key = eigval_key, eigvec_key = eigvec_key, out_key=out_key) # eigval_key, eigvec_key and out_key are not used 
		data.obsm[out_key] = result.values


	def run_palantur(self, data):


if __name__=="__main__":
	data_path = ... 
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	pw = PalantirWrapper()
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)	

	np,.random.seed(seed)
	starting_cell = np.random.choice(data.obs[data.obs["rna:celltype"]=="RG, Astro, OPC"].index, size = 1, replace=False)
	print(f"Starting cell: {starting_cell}") 
	pr_res = pw.run_palantir()

