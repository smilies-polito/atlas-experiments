import os
import unittest
import muon as mu
import pandas as pd
from anndata import AnnData
from muon import MuData
from atlas import ATLAS, PalantirWrapper, PseudotimeKernelWrapper


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

class TestInit(unittest.TestCase):
	def test_unavailable_method(self):
		mudata = MuData({"rna":rna, "activity":activity})
		with self.assertRaises(ValueError):
			atlas = ATLAS(mudata = mudata, method = "UNAVAILABLE METHOD")
	
	def test_palantir(self):
		mudata = MuData({"rna":rna, "activity":activity})
		atlas = ATLAS(mudata=mudata, 
				method="palantir", 
				fragment_path = fragment_path, 
				other="unused parameter")
		self.assertIsInstance(atlas._impl, PalantirWrapper)
		self.assertTrue(atlas._impl.use_activity)

	def test_pseudotime_kernel(self):
		mudata = MuData({"rna":rna, "activity":activity})
		atlas = ATLAS(mudata = mudata, 
				method="pseudotime-kernel", 
				fragment_path = fragment_path,
				pseudotime_key = "pseudotime", 
				connectivity_key = "wnn_connectivites", 
				backward=True, 
				other="unused parameter")
		self.assertIsInstance(atlas._impl, PseudotimeKernelWrapper)
		self.assertTrue(atlas._impl.use_activity)
		self.assertEqual(atlas._impl.pseudotime_key, "pseudotime")
		self.assertTrue(atlas._impl.backward)

class TestGetData(unittest.TestCase):
	def test_get_data(self):
		mudata = MuData({"rna":rna, "activity":activity})
		atlas = ATLAS(mudata=mudata, 
				method="palantir", 
				fragment_path = fragment_path, 
				other="unused parameter")
		newdata = atlas.get_data()
		self.assertEqual(mudata, newdata)	


if __name__ == "__main__":
	unittest.main()
