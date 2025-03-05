import os
import muon as mu
import numpy as np
import pandas as pd
import unittest
from core.palantirModel.palantir_plots import plot_pseudotime, plot_entropy, plot_probabilities, plot_palantir_results, pseudotime_correlation, entropy_correlation
from core.palantirModel.palantir_utils import _check_keys
from core.utils import simple_scatter, simple_heatmap

class Test_Keys(unittest.TestCase):

	def test_keys_mudata(self):
		data = mu.read_h5mu(os.path.join(os.getcwd(), "scMultiSim", "data.h5mu"))
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
		data = mu.read_h5mu(os.path.join(os.getcwd(), "scMultiSim", "data.h5mu"))
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
		data = mu.read_h5mu(os.path.join(os.getcwd(), "scMultiSim", "data.h5mu"))
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

	#TESTING TESTS
#	unittest.main()
	
	# TESTING PLOTS
	#color = pd.Series(np.random.uniform(0, 1, 120))
	#embedding = np.random.uniform(1, 100, (120,2))
	save = True
	saving_folder = os.path.join(os.getcwd(), "figure")

	#plot_pseudotime(pseudotime = color, embedding=embedding, save=save, saving_path = saving_folder)
	#plot_entropy(entropy = color, embedding=embedding, save=save, saving_path = saving_folder)

	#fates = pd.DataFrame({})
	#terminal_states = fates.columns.values
	#terminal_indices = np.where(fates.index.isin(terminal_states))
	#terminal_locations = None
	#plot_probabilities(fates=fates, terminal_states = terminal_states, terminal_locations = terminal_locations, embedding= embedding, save=save, saving_path = saving_folder)

	#fates = pd.DataFrame({"cell1": np.random.uniform(0,1,120)})
	#fates["cell2"] = 1 - fates["cell1"] 
	#fates.index = ["cell" + str(i) for i in range(120)]
	#terminal_states = fates.columns
	#terminal_indices = np.where(fates.index.isin(terminal_states))[0]
	#terminal_locations = embedding[terminal_indices, :]
	#plot_probabilities(fates = fates, terminal_states = terminal_states, terminal_locations = terminal_locations, embedding=embedding, save=save, saving_path= saving_folder)

	data = mu.read_h5mu(os.path.join(os.getcwd(), "scMultiSim", "data.h5mu"))
	modality_key = "rna"
	pseudo_time_key = "palantir_pseudotime"
	embedding_key = "X_umap"
	entropy_key = "entropy"
	fate_prob_key = "fates"
	
	data[modality_key].obs[pseudo_time_key]=np.random.uniform(0,1, len(data.obs_names))
	data[modality_key].obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
#	data.obs[pseudo_time_key]=np.random.uniform(0,1, len(data.obs_names))
#	data.obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
#	cells = np.random.choice(data.obs_names, 2)
#	fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
#	data[modality_key].obsm[fate_prob_key] = fates
#	data.obsm[fate_prob_key] = fates

#	plot_palantir_results(data, modality_key = modality_key, embedding_key = embedding_key, pseudo_time_key = pseudo_time_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key, save=save, saving_path = saving_folder)

#	pseudotime_correlation(data= data, modality_key=modality_key, true_key = "pseudotime", infer_key = pseudo_time_key, save=True, saving_path = saving_folder, group_key = "pop")
	entropy_correlation(data=data, modality_key=modality_key, pseudo_time_key = "pseudotime", entropy_key = entropy_key, save=True, saving_path = saving_folder, group_key= "pop")



#	correlations1 = np.random.uniform(0, 1, (3,4))
#	correlations2 = correlations1 + 1
#	mask = np.random.choice([0,1], (3,4)).astype(bool)
#	path1 = os.path.join(saving_folder, "heatmap1.png")
#	path2 = os.path.join(saving_folder, "heatmap2.png")
#	cmap_range = (0,2)	
#	simple_heatmap(correlations1, mask, annot=True, saving_path = path1, cmap_range=cmap_range, title = "Hetmap1", xlabel= "X", ylabel="Y", cmap_label="label")
#	simple_heatmap(correlations2, mask, annot=True, saving_path = path2, cmap_range= cmap_range) 



