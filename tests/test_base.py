import os
import unittest
import muon as mu
import numpy as np
import scanpy as sc
import pandas as pd
from atlas import Base
from muon import MuData
from anndata import AnnData

fragment_path = os.path.join(os.getcwd(), "tests", "dummy_fragments.tsv")
rna = AnnData(X= pd.read_csv(os.path.join(os.getcwd(), "tests", "dummy_rna.tsv"),
				sep="\t",
				header=0,
				index_col=0).T.values)
activity = AnnData(X= pd.read_csv(os.path.join(os.getcwd(), "tests", "dummy_activity.tsv"),
				sep="\t",
				header=0,
				index_col=0).values)
atac = AnnData(X= pd.read_csv(os.path.join(os.getcwd(), "tests", "dummy_atac.tsv"),
				sep="\t",
				header=0,
				index_col=0).T.values)

wrong_shape_activity = activity[:10, :10].copy()

class DummyClass(Base):
	def run(self, message:str)->None:
		return message


class TestInit(unittest.TestCase):
	def test_missing_rna(self):
		with self.assertRaises(KeyError):
			DummyClass(mudata=MuData({
						"activity":activity,
						"atac": atac
				}))
		
	def test_missing_atac_and_activity(self):
		with self.assertRaises(KeyError):
			DummyClass(mudata=MuData({"rna":rna}))

	def test_wrong_shape_activity(self):
		with self.assertRaises(ValueError):
			DummyClass(mudata=MuData({
						"rna":rna,
						"activity": wrong_shape_activity
					}))

	def test_double_modality_warning(self):
		with self.assertWarns(UserWarning):
			DummyClass(mudata=MuData({
						"rna":rna,
						"activity": activity,
						"atac": atac
					}))

	def test_case_insensitive_modality_name(self):
		mudata = MuData({"RNA":rna, "ATac": atac, "aCTIvity": activity})
		dc = DummyClass(mudata=mudata)
		self.assertEqual(dc.rna_key, "RNA")
		self.assertIsNone(dc.atac_key)
		self.assertEqual(dc.activity_key, "aCTIvity")
		self.assertTrue(dc.use_activity)

	def test_fragments_without_atac(self):
		mudata = MuData({"rna": rna, "activity": activity})
		with self.assertWarns(UserWarning):
			dc = DummyClass(mudata=mudata, fragment_path = fragment_path)
			
	def test_locate_fragments(self):
		mudata = MuData({"rna":rna, "aTac":atac.copy()})
		dc = DummyClass(mudata=mudata, fragment_path = fragment_path)
		self.assertIn("atac", mudata.mod.keys())
		self.assertEqual("atac", dc.atac_key)
		self.assertIn("files", mudata["atac"].uns)
		self.assertIn("fragments", mudata["atac"].uns["files"])
		self.assertEqual(fragment_path, mudata["atac"].uns["files"]["fragments"])


class TestRun(unittest.TestCase):
	def test_run(self):
		message = "ciao"
		mudata = MuData({"rna":rna, "activity":activity})
		dc = DummyClass(mudata = mudata)
		result = dc.run(message=message)
		self.assertEqual(message, result)

class TestGetData(unittest.TestCase):
	def test_get_mudata(self):
		mudata = MuData({"rna":rna, "activity":activity})
		dc = DummyClass(mudata = mudata)
		result = dc.get_data()
		self.assertEqual(mudata, result)

class TestPreprocessing(unittest.TestCase):
	def test_activity_available(self):		
		n_pcs_rna, n_pcs_act, knn_rna, knn_act, n_neighbors, = 10, 5, 30, 30, 30
		n_multineighbors, n_bandwidth_neighbors = 200, 20
		mudata = MuData({"rna":rna, "activity":activity})
		dc = DummyClass(mudata = mudata)
		sc.pp.normalize_total(mudata["rna"])
		sc.pp.log1p(mudata["rna"])
		sc.pp.highly_variable_genes(mudata["rna"], n_top_genes=50)
		sc.pp.normalize_total(mudata["activity"])
		dc.preprocessing(n_pcs_rna= n_pcs_rna, 
				n_pcs_act= n_pcs_act, 
				knn_rna= knn_rna, 
				knn_act= knn_act, 
				n_neighbors= n_neighbors,
				n_multineighbors = n_multineighbors,
				n_bandwidth_neighbors= n_bandwidth_neighbors)
		#test creazione dei single-modality neighbors e pca 
		self.assertIn("X_pca", mudata["rna"].obsm)	
		self.assertIn("X_pca", mudata["activity"].obsm)	
		self.assertIn("distances", mudata["rna"].obsp)	
		self.assertIn("distances", mudata["activity"].obsp)	
		self.assertIn("connectivities", mudata["rna"].obsp)	
		self.assertIn("connectivities", mudata["activity"].obsp)	
		self.assertIn("wnn_distances", mudata.obsp)	
		self.assertIn("wnn_connectivities", mudata.obsp)	
		self.assertEqual(n_neighbors, mudata.uns["wnn"]["params"]["n_neighbors"])

	def test_fragments_not_available(self):
		mudata = MuData({"rna":rna, "atac":atac.copy()})
		dc = DummyClass(mudata = mudata, fragment_path= None)
		with self.assertRaises(KeyError):
			dc.preprocessing()

	def test_features_is_None(self):
		mudata = MuData({"rna":rna, "atac":atac.copy()})
		dc = DummyClass(mudata = mudata, fragment_path=fragment_path)
		with self.assertRaises(ValueError):
			dc.preprocessing()
			


if __name__=="__main__":
	unittest.main()
			
