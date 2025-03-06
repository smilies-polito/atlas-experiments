import os
import muon as mu
import numpy as np
import pandas as pd
import unittest
from core.palantirModel.utils import _check_keys, _save_results

class Test(unittest.TestCase):

	def test_keys_mudata(self):
		data = mu.read_h5mu( ... )
		modality_key = None
		pseudotime_key = "palantir_pseudotime"
		embedding_key = "X_umap"
		entropy_key = "entropy"
		fate_prob_key = "fates"
		obs_key = "keyword"
		
		data.obs[pseudotime_key]=np.random.uniform(0,1, len(data.obs_names))
		data.obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
		data.obs[obs_key] = True
		cells = np.random.choice(data.obs_names, 2)
		fates = pd.DataFrame( np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
		data.obsm[fate_prob_key] = fates

		with self.assertRaises(KeyError): 
				_check_keys(data, modality_key= "modality_key", pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
				_check_keys(data, modality_key = modality_key, pseudo_time_key = "pseudo_time_key", embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key)
		
		with self.assertRaises(KeyError):
			 _check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = "embedding_key", entropy_key = entropy_key, fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = "entropy_key", fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = "fate_prob_key")
			
		with self.assertRaises(KeyError):
			_check_keys(data, obs_key= "obs_key")

		try:
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key, obs_key=obs_key)
		except:
			print("Wrong exception raised")



	def test_keys_modality(self):
		data = mu.read_h5mu( ... )
		modality_key = "rna"
		pseudotime_key = "palantir_pseudotime"
		embedding_key = "X_umap"
		entropy_key = "entropy"
		fate_prob_key = "fates"
		obs_key = "keyword"
		
		data[modality_key].obs[pseudotime_key]=np.random.uniform(0,1, len(data.obs_names))
		data[modality_key].obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
		data[modality_key].obs[obs_key] = True
		cells = np.random.choice(data.obs_names, 2)
		fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
		data[modality_key].obsm[fate_prob_key] = fates

		with self.assertRaises(KeyError): 
				_check_keys(data, modality_key= "modality_key", pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
				_check_keys(data, modality_key = modality_key, pseudo_time_key = "pseudo_time_key", embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key)
		
		with self.assertRaises(KeyError):
			 _check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = "embedding_key", entropy_key = entropy_key, fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = "entropy_key", fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = "fate_prob_key")
			
		with self.assertRaises(KeyError):
			_check_keys(data, obs_key = "obs_key")

		try:
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key, obs_key=obs_key)
		except:
			print("Wrong Exception raised")


	def test_keys_adata(self):
		data = mu.read_h5mu( ... )
		modality_key = "rna"
		pseudotime_key = "palantir_pseudotime"
		embedding_key = "X_umap"
		entropy_key = "entropy"
		fate_prob_key = "fates"
		obs_key = "keyword"
		
		data[modality_key].obs[pseudotime_key]=np.random.uniform(0,1, len(data.obs_names))
		data[modality_key].obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
		data[modality_key].obs[obs_key] = True
		cells = np.random.choice(data.obs_names, 2)
		fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
		data[modality_key].obsm[fate_prob_key] = fates
		data = data[modality_key]			
	
		with self.assertRaises(KeyError):
				_check_keys(data, modality_key = modality_key, pseudo_time_key = "pseudo_time_key", embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key)
		
		with self.assertRaises(KeyError):
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = "entropy_key", fate_prob_key = fate_prob_key)

		with self.assertRaises(KeyError):
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = embedding_key, entropy_key = entropy_key, fate_prob_key = "fate_prob_key")

		with self.assertRaises(KeyError):
			_check_keys(data, obs_key = "obs_key")
			
		try:
			_check_keys(data, modality_key = modality_key, pseudo_time_key = pseudotime_key, embedding_key = "embedding_key", entropy_key = entropy_key, fate_prob_key = fate_prob_key)
		except:
			print("Wrong Exception raised")



if __name__=="__main__":
	unittest.main()
	
	# TESTING SAVE RESULTS MUON DATA
	data = mu.read_h5mu( ... )
	modality_key = None
	pseudo_time_key = "palantir_pseudotime"
	embedding_key = "X_umap"
	entropy_key = "entropy"
	fate_prob_key = "fates"
	saving_path = os.path.join(os.getcwd(), "palantir_muon")
	
	data.obs[pseudo_time_key]=np.random.uniform(0,1, len(data.obs_names))
	data.obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
	cells = np.random.choice(data.obs_names, 2)
	fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
	data.obsm[fate_prob_key] = fates
	
	# Test with everything
	_save_results(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key = pseudo_time_key, fate_prob_key = fate_prob_key, saving_path=os.path.join(saving_path, "all.tsv"), group_key = "rna:pop", true_pseudotime="rna:pseudotime")
	
	# Test with only mandatory keys (no kwargs)
	_save_results(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key = pseudo_time_key, fate_prob_key = fate_prob_key, saving_path=os.path.join(saving_path, "partial.tsv"))

	# Test append with no kwargs
	_save_results(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key = pseudo_time_key, fate_prob_key = fate_prob_key, saving_path=os.path.join(saving_path, "partial.tsv"))



	# TESTING SAVE RESULTS ANNDATA
	data = mu.read_h5mu( ... )
	modality_key = "rna"
	pseudo_time_key = "palantir_pseudotime"
	embedding_key = "X_umap"
	entropy_key = "entropy"
	fate_prob_key = "fates"
	saving_path = os.path.join(os.getcwd(), "palantir_rna")

	data[modality_key].obs[pseudo_time_key]=np.random.uniform(0,1, len(data.obs_names))
	data[modality_key].obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
	cells = np.random.choice(data.obs_names, 2)
	fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index= data.obs_names)
	data[modality_key].obsm[fate_prob_key] = fates

	# Test with everything
	_save_results(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key = pseudo_time_key, fate_prob_key = fate_prob_key, saving_path=os.path.join(saving_path, "all.tsv"), group_key = "pop", true_pseudotime="pseudotime")
	
	# Test with only mandatory keys (no kwargs)
	_save_results(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key = pseudo_time_key, fate_prob_key = fate_prob_key, saving_path=os.path.join(saving_path, "partial.tsv"))

	# Test append with no kwargs
	_save_results(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key = pseudo_time_key, fate_prob_key = fate_prob_key, saving_path=os.path.join(saving_path, "partial.tsv"))
