import os, math
import unittest 
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from atlas import pearson_entropy_pseudotime, spearman_entropy_pseudotime, fate_concentration_index, terminal_state_silhouette, _hard_ai, _soft_ai, _hard_bi, _soft_bi, terminal_pseudotime_enrichment_score

class TestTerminalPseudotimeEnrichmentScore(unittest.TestCase):
	def setUp(self):
		self.pseudotime = pd.Series([0.1, 0.1, 0.2, 0.9, 0.9, 0.6, 0.7, 0.8],
				 index = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c9"])
	
	def test_empty_terminal(self):
		with self.assertRaises(ValueError):
			terminal_pseudotime_enrichment_score(terminal_states = {},
								pseudotime = self.pseudotime)	
	def test_single_terminal_state_rank(self):
		pseudotime = pd.Series([0.1, 0.5, 0.9], index = ["c1", "c2", "c3"])
		ts = {"t1": ["c3"]}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime,
								rank= True)
		expected = 0.5 
		self.assertAlmostEqual(expected, tpes) 

	def test_single_terminal_state_raw(self):
		pseudotime = pd.Series([0.1, 0.5, 0.9], index = ["c1", "c2", "c3"])
		ts = {"t1": ["c3"]}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime)
		expected = 0.4
		self.assertAlmostEqual(expected, tpes) 

	def test_multicell_terminal(self):
		pseudotime = pd.Series([0.1, 0.4, 0.8, 0.9], index = ["c1", "c2", "c3", "c4"])
		ts = {"t1": ["c3" ,"c4"]}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime)
		expected = 0.85 - 0.6
		self.assertAlmostEqual(expected, tpes) 

	def test_multiple_terminal(self):
		pseudotime = pd.Series([0.1, 0.3, 0.7, 0.9], index = ["c1", "c2", "c3", "c4"])
		ts = {"t1": ["c3"], "t2": ["c4"]}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime)
		global_mean = 0.5 
		expected = ((0.7 - global_mean) + (0.9 - global_mean)) / 2
		self.assertAlmostEqual(expected, tpes) 

	def test_rank_based_invariance_to_scaling(self):
		pseudotime1 = pd.Series([0.1, 0.5, 0.9], index = ["c1", "c2", "c3"])
		pseudotime2 = pseudotime1 * 100
		ts = {"t1": ["c3"]}
		tpes1 = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime1,
								rank = True)
		
		tpes2 = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime2,
								rank = True)
		self.assertAlmostEqual(tpes1, tpes2)

	def test_rank_with_ties(self):
		pseudotime = pd.Series([0.2, 0.5, 0.5, 0.9], index = ["c1", "c2", "c3", "c4"])
		ts = {"t1": ["c2", "c3"]}
		expected = 0.0 
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime,
								rank = True)
		self.assertAlmostEqual(expected, tpes)

	def test_single_cell_dataset(self):
		pseudotime = pd.Series([0.2], index = ["c1"])
		ts = {"t1": ["c1"]}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime,
								rank = True)
		self.assertAlmostEqual(tpes, 0.0)

	def test_non_informative_ts(self):
		pseudotime = pd.Series(np.linspace(0,1, 100), index= [f"c{i}" for i in range(100)])
		ts = {"t1": pseudotime.sample(20, random_state=0).index.tolist()}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime)
		self.assertTrue(abs(tpes) < 0.1)
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime, 
								rank = True)
		self.assertTrue(abs(tpes) < 0.1)

	def test_early_cells(self):
		pseudotime = pd.Series(np.linspace(0,1, 100), index= [f"c{i}" for i in range(100)])
		ts = {"t1": pseudotime.nsmallest(10).index.tolist()}
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime)
		self.assertLess(tpes, 0.0)
		tpes = terminal_pseudotime_enrichment_score(terminal_states=ts, 
								pseudotime = pseudotime, 		
								rank = True)
		self.assertLess(tpes, 0.0)
	
		
class TerminalStateSilhouette(unittest.TestCase):
	def test_no_fates(self):
		pseudotime = pd.Series([0.2, 0.4, 0.6, 0.8])
		with self.assertRaises(ValueError):
			terminal_state_silhouette(fates=pd.DataFrame(index=pseudotime.index))

	def test_hard_no_pseudotime(self):
		fates = pd.DataFrame([[0,1],[1,0]])
		with self.assertRaises(ValueError):
			terminal_state_silhouette(fates, soft_assignment=False, pseudotime=None)

	def test_nan(self):
		fates = pd.DataFrame([[1,0], [np.nan, np.nan], [0,1]])
		with self.assertRaises(ValueError):
			S = terminal_state_silhouette(fates)

	def test_single_terminal(self):
		fates = pd.DataFrame({"A": [1,1]})
		with self.assertWarns(UserWarning):
			result = terminal_state_silhouette(fates)
		self.assertEqual(result, 0)

	def test_soft_assignment_basic(self):
		fates = pd.DataFrame([[1,0], [0.9,0.1], [0.1, 0.9], [0,1]],
					index = ["c1", "c2", "c3", "c4"], 
					columns = ["A", "B"])
		S = terminal_state_silhouette(fates, soft_assignment=True)
		self.assertTrue(-1 <= S <= 1)

	def test_hard_assignment_basic(self):
		fates = pd.DataFrame([[1,0], [1,0], [0,1], [0,1]],
					index = ["c1", "c2", "c3", "c4"], 
					columns = ["A", "B"])
		pseudotime = pd.Series([0.2, 0.4, 0.6, 0.8], index = fates.index)
		S = terminal_state_silhouette(fates, pseudotime=pseudotime, soft_assignment=False)
		self.assertTrue(-1 <= S <= 1)
		

	def test_hard_ai(self):	
		# one cell into another terminal
		f = pd.DataFrame({"A": [0.9, 0.8, 0.1], "B": [0.1, 0.2, 0.9]}, 
				index = ["c1","c2","c3"])
		D = pd.DataFrame([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
				index = ["c1", "c2", "c3"],
				columns = ["c1", "c2", "c3"])
		expected = pd.Series({"c1": 1.0, "c2": 1.0, "c3": np.nan})
		result = _hard_ai(f, D)
		assert_series_equal(result, expected)	
		# all cells committed to the same terminal
		f = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, 
				index = ["c1", "c2", "c3"])
		result = _hard_ai(f, D)
		expected = pd.Series({"c1": (1.0 + 2.0)/2,
					"c2": (1.0 + 3.0)/2,
					"c3": (2.0 + 3.0)/2})
		assert_series_equal(result, expected)
		# all cells committed to a different terminal 
		f = pd.DataFrame({ "A": [1.0, 0.0, 0.0], "B":[0.0, 1.0, 0.0], "C":[0.0,0.0,1.0]},
				index = ["c1","c2", "c3"])
		result = _hard_ai(f, D)
		expected = pd.Series({ "c1":np.nan, "c2": np.nan, "c3": np.nan})
		assert_series_equal(result, expected)

	def test_soft_ai(self):
		# one cell into another terminal
		f = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]}, 
				index = ["c1","c2","c3"])
		D = pd.DataFrame([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]],
				index = ["c1", "c2", "c3"],
				columns = ["c1", "c2", "c3"])
		expected = pd.Series({"c1":2.0, "c2": 2.0, "c3":np.nan})
		result = _soft_ai(f, D)
		assert_series_equal(expected, result)
		# all cells committed to a different terminal 
		f = pd.DataFrame({ "A": [1.0, 0.0, 0.0], "B":[0.0, 1.0, 0.0], "C":[0.0, 0.0, 1.0]},
				index = ["c1","c2", "c3"])
		result = _soft_ai(f, D)
		expected = pd.Series({"c1":np.nan, "c2": np.nan, "c3": np.nan})
		assert_series_equal(result, expected)
		# all cells committed to the same terminal
		f = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, 
				index = ["c1", "c2", "c3"])
		result = _soft_ai(f, D)
		expected = pd.Series({"c1": 3.0, "c2": 4.0, "c3": 5.0})
		assert_series_equal(result, expected)
		# test general case 
		f = pd.DataFrame({"A":[0.8, 0.6, 0.0], "B": [0.2, 0.4, 1.0]}, 
				index = ["c1", "c2", "c3"])
		D = pd.DataFrame([[0.0, 1.0, 3.0], [1.0, 0.0, 5.0], [3.0, 5.0, 0.0]],
				index = ["c1", "c2", "c3"],
				columns = ["c1", "c2", "c3"])
		expected_c1 = (0.56 * 1.0 + 0.2 * 3.0) / (0.56 + 0.2)
		expected_c2 = (0.56 * 1.0 + 0.4 * 5.0) / (0.56 + 0.4)
		expected_c3 = (0.2 * 3.0 + 0.4 * 5.0) / (0.2 + 0.4)
		expected = pd.Series({"c1": expected_c1, "c2": expected_c2, "c3": expected_c3})
		result = _soft_ai(f,D)
		assert_series_equal(result, expected)

	def test_hard_bi(self):
		D = pd.DataFrame([[0,2,3],[2,0,1],[3,1,0]],
				index = ["c1", "c2", "c3"],
				columns = ["c1", "c2", "c3"])
		# all cells committed to different states
		f = pd.DataFrame(np.eye(3), index = ["c1", "c2", "c3"], columns=[0,1,2])
		bi = _hard_bi(f,D)
		expected = pd.Series([2.0, 1.0, 1.0], index=["c1", "c2", "c3"])
		assert_series_equal(bi, expected)
		# all cells committed to the same fate
		f = pd.DataFrame([[1.0, 1.0, 1.0],[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
				index = ["c1", "c2", "c3"],
				columns = [0,1,2])
		bi = _hard_bi(f, D)
		expected = pd.Series({"c1":np.nan, "c2":np.nan, "c3":np.nan})
		assert_series_equal(bi, expected)
		# two cluster and one isolated cell
		f = pd.DataFrame({"A":[1.0, 1.0, 0.0], "B":[0.0, 0.0, 1.0]}, 
				index = ["c1", "c2", "c3"])
		bi = _hard_bi(f,D)
		expected = pd.Series({"c1":3.0, "c2":1.0, "c3": 2.0})
		assert_series_equal(bi, expected)
		# three clusters one is empty 
		f = pd.DataFrame([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
				index = ["c1", "c2", "c3"],
				columns = [0,1,2])
		bi = _hard_bi(f,D)
		expected = pd.Series({"c1":3.0, "c2":1.0, "c3": 2.0})
		assert_series_equal(bi, expected)
		
	
	def test_soft_bi(self):
		D = pd.DataFrame([[0,2,3],[2,0,1],[3,1,0]],
				index = ["c1", "c2", "c3"],
				columns = ["c1", "c2", "c3"])
		# all cells committed to different states
		f = pd.DataFrame(np.eye(3), index = ["c1", "c2", "c3"], columns=[0,1,2])
		bi = _soft_bi(f,D)
		expected = pd.Series([2.5, 1.5, 2.0], index=["c1", "c2", "c3"])
		assert_series_equal(bi, expected)
		# all cells committed to the same fate
		f = pd.DataFrame({"A":[1.0, 1.0, 1.0], "B":[0.0, 0.0, 0.0],"C": [0.0, 0.0, 0.0]},
				index = ["c1", "c2", "c3"])
		bi = _soft_bi(f, D)
		expected = pd.Series({"c1":np.nan, "c2":np.nan, "c3":np.nan})
		assert_series_equal(bi, expected)
		# two cluster and one isolated cell
		f = pd.DataFrame({"A":[1.0, 1.0, 0.0], "B":[0.0, 0.0, 1.0]}, 
				index = ["c1", "c2", "c3"])
		bi = _soft_bi(f,D)
		expected = pd.Series({"c1":3.0, "c2":1.0, "c3": 2.0})
		assert_series_equal(bi, expected)
		
		
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
