import anndata
import unittest
import numpy as np 
import pandas as pd

from math import e
from transitionMatrix import Similarity, SimilarityComputer, SimilarityWrapper, Cosine, Correlation, DotProduct

class Test(unittest.TestCase):

    def test_similarityWrapper_correlation(self):
        sw = SimilarityWrapper.create('correlation')
        self.assertIsInstance(sw, Correlation)
        self.assertTrue(sw._center_mean)
        self.assertTrue(sw._scale_by_norm)

    def test_similarityWrapper_cosine(self):
        sw = SimilarityWrapper.create('cosine')
        self.assertIsInstance(sw, Cosine)
        self.assertFalse(sw._center_mean)
        self.assertTrue(sw._scale_by_norm)

    def test_similarityWrapper_dotprod(self):
        sw = SimilarityWrapper.create('dot')
        self.assertIsInstance(sw, DotProduct)
        self.assertFalse(sw._center_mean)
        self.assertFalse(sw._scale_by_norm)

    def test_similarityComputer_softmax(self):
        sc = SimilarityComputer()
        self.assertFalse(sc._center_mean, sc._scale_by_norm)

        numerator = np.array([1,2])
        softmax_scale = 1.0
        result, _= sc.softmax(numerator, softmax_scale)
        
        self.assertEquals(len(result), 2)
        self.assertAlmostEquals(result[-1], 1/(e**(-1) + 1))

    def test_similarityComputer_softmax_masked(self):
        sc = SimilarityComputer()
        self.assertFalse(sc._center_mean, sc._scale_by_norm)

        numerator = np.array([1,2])
        denom = np.array([0,1])
        mask = denom == 0 #[True, False]
        denom[mask] = 1 #[1,1]
        softmax_scale = 1.0
        result, _= sc.softmax_masked(numerator/denom, mask, softmax_scale)

        self.assertEquals(len(result), 2)
        self.assertAlmostEquals(result[0], 0)


    def test_call_DotProduct(self): #OK
        sw = SimilarityWrapper.create('dot')
        self.assertFalse(sw._center_mean)
        self.assertFalse(sw._scale_by_norm)

        X = np.array([[1,2,3],[1,2,1],[1,2,1]])
        v = np.array([1,1,1])
        softmax_scale = 0.5

        probs, logits = sw(v, X, softmax_scale)
        self.assertEquals(probs.shape[0], 3)
        self.assertEquals(logits.shape[0], 3)
        self.assertEquals(logits[0], 6)
        self.assertEquals(probs[0], 1/(1+2*e**(-1)))


    def test_call_Correlation(self): #OK 
        sw = SimilarityWrapper.create('correlation')
        self.assertTrue(sw._center_mean)
        self.assertTrue(sw._scale_by_norm)

        X = np.array([[1,2,3],[1,2,1],[1,0,0]], dtype=np.float64)
        v = np.array([1,2,1], dtype=np.float64)
        softmax_scale = 0.5

        probs, logits = sw(v, X, softmax_scale)
        self.assertEqual(probs.shape[0], 3)
        self.assertEquals(logits.shape[0], 3)
        self.assertEquals(logits[0], 0.)
        self.assertEquals(logits[1], 1.)
        self.assertEquals(logits[2], -0.5)
        self.assertAlmostEquals(probs[0], 0.29175596)
        self.assertAlmostEquals(probs[2], 0.22721977)
        self.assertAlmostEquals(np.sum(probs), 1.0)
        


if __name__=='__main__':
    unittest.main()