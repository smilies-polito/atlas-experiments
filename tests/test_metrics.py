import os, math
import unittest 
import numpy as np
import pandas as pd
from atlas import pearson_entropy_pseudotime, spearman_entropy_pseudotime, fate_concentration_index

class TestFateConcentration(unittest.TestCase):
	def setUp(self):
		n_over = 600
		rng = np.random.default_rng(42)
		self.pseudotime = pd.Series(np.linspace(0,1, n_over))
		self.fully_committed = pd.DataFrame({"A": np.zeros(n_over), 
							"B": np.ones(n_over)})
		self.uncorrelated = pd.DataFrame(rng.dirichlet(alpha=[1,1,1], size=n_over),
						columns = ["A","B","C"])
		progressive = [ [0.33, 0.33, 0.34] if t < 0.5 
							else [t, 1-t, 0] 
							for t in self.pseudotime]
		self.progressive = pd.DataFrame(progressive, columns = ["A", "B", "C"])


	def test_shape_mismatch(self):	
		with self.assertRaises(ValueError):
			fate_concentration_index(fates = self.progressive, pseudotime=self.pseudotime.iloc[:10])
	
	def test_no_association(self):
		rho, pval, ci, _ = fate_concentration_index( fates=self.uncorrelated,
								pseudotime = self.pseudotime)
		self.assertAlmostEqual(rho, 0.0, delta=0.15)

	def test_fully_committed(self):
		rho, _, __, concentration = fate_concentration_index(fates=self.fully_committed, 
									pseudotime = self.pseudotime)
		self.assertTrue(np.allclose(concentration.values, 1.0))
		self.assertTrue(np.isnan(rho) or abs(rho) < 1e-6)	
	
	def test_monotonic(self):
		rho, pval, _, __ = fate_concentration_index(fates = self.progressive,
							pseudotime = self.pseudotime)
		self.assertGreater(rho, 0.5)
		self.assertLess(pval, 0.05)

	def test_ci_not_None(self):
		_, __, ci, ___ = fate_concentration_index(fates = self.progressive.iloc[:100, :], 
								pseudotime = self.pseudotime.iloc[:100])	
		self.assertIsNotNone(ci)
		self.assertTrue(hasattr(ci, "low"))
		self.assertTrue(hasattr(ci, "high"))
	
	def test_reindex(self):
		x = self.pseudotime.copy()
		x_index = x.index.to_numpy().copy()
		x_index[0], x_index[30] = x_index[30], x_index[0]
		x = x.loc[x_index]
		r, pval, __, ___ = fate_concentration_index(fates = self.progressive,
							pseudotime = x)
		self.assertGreater(r, 0.5)
		self.assertLess(pval, 0.05)

	def test_single_fate(self):
		rho, ___, __, concentration = fate_concentration_index(fates = pd.DataFrame({"A":np.ones(len(self.pseudotime))}),
								pseudotime = self.pseudotime)
		self.assertTrue(np.allclose(concentration, 1.0))
		self.assertTrue(np.isnan(rho) or abs(rho) < 1e-6)

	def test_no_fates(self):
		with self.assertRaises(ValueError):
			fate_concentration_index(fates = pd.DataFrame(index = self.pseudotime.index),
								pseudotime = self.pseudotime)
		

class TestSpearmanEntropyPseudotime(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(42)
		self.x_over = pd.Series(rng.uniform(0,1, size=(1000)))
		self.x_under = pd.Series(rng.uniform(0,1, size=(300)))
		self.y_over_pos = self.x_over.copy()
		self.y_over_neg = - self.x_over.copy()
		self.y_under = self.x_under.copy() 
		self.y_non_linear = self.x_over ** 3

	def test_input_shape(self):
		with self.assertRaises(ValueError):
			spearman_entropy_pseudotime(self.x_over, self.x_under.copy())
	
	def test_nans(self):
		x, y = self.x_under.copy(), self.y_under.copy()
		x[0], y[0] = np.nan, np.nan
		with self.assertRaises(ValueError):
			spearman_entropy_pseudotime(x, self.y_under)
		with self.assertRaises(ValueError):
			spearman_entropy_pseudotime(self.x_under, y)
	

	def test_reindex(self):
		x = self.x_over.copy()
		y = x.copy()
		x_index = x.index.to_numpy().copy()
		x_index[0], x_index[30] = x_index[30], x_index[0]
		x = x.loc[x_index]
		r, p, ci = spearman_entropy_pseudotime(x, y)
		self.assertAlmostEqual(r, 1.0)

	def test_ci_None(self):
		r, p, ci = spearman_entropy_pseudotime(self.x_over, self.y_over_pos)
		self.assertIsNone(ci)

	def test_ci_not_None(self):
		r, p, ci = spearman_entropy_pseudotime(self.x_under, self.y_under)
		self.assertIsNotNone(ci)
		self.assertTrue(hasattr(ci, "low"))
		self.assertTrue(hasattr(ci, "high"))

	def test_seed_reproducibility(self):
		r1, p1, ci1 = spearman_entropy_pseudotime(self.x_under, self.y_under, seed=123)
		r2, p2, ci2 = spearman_entropy_pseudotime(self.x_under, self.y_under, seed=123)
		self.assertAlmostEqual(r1, r2)
		self.assertAlmostEqual(p1, p2)
		if math.isnan(ci1.low):
			self.assertTrue(math.isnan(ci2.low))
			self.assertTrue(math.isnan(ci2.high))
		else:
			self.assertAlmostEqual(ci1.low, ci2.low)
			self.assertAlmostEqual(ci1.high, ci2.high)

	def test_perfect_positive_correlation(self):
		r, p, ci = spearman_entropy_pseudotime(self.x_over, self.y_over_pos)
		self.assertAlmostEqual(r, 1.0)

	def test_monotonic_non_linear_relationship(self):
		r, p, ci = spearman_entropy_pseudotime(self.x_over, self.y_non_linear)	
		self.assertAlmostEqual(r, 1.0, places=6)


class TestPearsonEntropyPseudotime(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(42)
		self.x_over = pd.Series(rng.uniform(0,1, size=(1000)))
		self.x_under = pd.Series(rng.uniform(0,1, size=(300)))
		self.y_over_pos = self.x_over.copy()
		self.y_over_neg = - self.x_over.copy()
		self.y_under = self.x_under.copy()

	def test_input_shape(self):
		with self.assertRaises(ValueError):
			pearson_entropy_pseudotime(self.x_over, self.x_under.copy())

	def test_nans(self):
		x, y = self.x_under.copy(), self.y_under.copy()
		x[0], y[0] = np.nan, np.nan
		with self.assertRaises(ValueError):
			pearson_entropy_pseudotime(x, self.y_under)
		with self.assertRaises(ValueError):
			pearson_entropy_pseudotime(self.x_under, y)

	def test_reindex(self):
		x = self.x_over.copy()
		y = x.copy()
		x_index = x.index.to_numpy().copy()
		x_index[0], x_index[30] = x_index[30], x_index[0]
		x = x.loc[x_index]
		r, p, ci = pearson_entropy_pseudotime(x, y)
		self.assertAlmostEqual(r, 1.0)

	def test_perfect_positive_correlation(self):
		r, p, ci = pearson_entropy_pseudotime(self.x_over, self.y_over_pos)
		self.assertAlmostEqual(r, 1.0)

	def test_perfect_negative_correlation(self):
		r, p, ci = pearson_entropy_pseudotime(self.x_over, self.y_over_neg)
		self.assertAlmostEqual(r, -1.0)

	def test_ci_None(self):
		r, p, ci = pearson_entropy_pseudotime(self.x_over, self.y_over_pos)
		self.assertIsNone(ci)

	def test_ci_not_None(self):
		r, p, ci = pearson_entropy_pseudotime(self.x_under, self.y_under)
		self.assertIsNotNone(ci)
		self.assertTrue(hasattr(ci, "low"))
		self.assertTrue(hasattr(ci, "high"))

	def test_seed_reproducibility(self):
		r1, p1, ci1 = pearson_entropy_pseudotime(self.x_under, self.y_under, seed=123)
		r2, p2, ci2 = pearson_entropy_pseudotime(self.x_under, self.y_under, seed=123)
		self.assertAlmostEqual(r1, r2)
		self.assertAlmostEqual(p1, p2)
		if math.isnan(ci1.low):
			self.assertTrue(math.isnan(ci2.low))
			self.assertTrue(math.isnan(c12.high))
		else:
			self.assertAlmostEqual(ci1.low, ci2.low)
			self.assertAlmostEqual(ci1.high, ci2.high)

	


if __name__=="__main__":
	unittest.main()
