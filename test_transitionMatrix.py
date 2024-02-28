import anndata
import unittest
import numpy as np 
import pandas as pd

from math import e
from scipy.sparse import csr_matrix
from transitionMatrix import Similarity, SimilarityComputer, SimilarityWrapper, Cosine, Correlation, DotProduct
from transitionMatrix import Deterministic, TransitionMatrix

class Test(unittest.TestCase):
    obs_names = ['cell1', 'cell2', 'cell3', 'cell4', 'cell5']
    var_names = ['peak1', 'peak2', 'peak3', 'peak4']
    gene_names = ['gene0', 'gene1','gene2', 'gene3', 'gene4', 'gene5']
    X = np.random.uniform(size=(5,4))
    df = pd.DataFrame({'gene': ['gene5', 'gene2', 'gene0', 'gene1']}, index = var_names)
    velocities = pd.DataFrame(np.random.uniform(size=(5,6)), index = obs_names, columns = gene_names)
    connectivities = csr_matrix(np.array([[0,1,1,0,0], [1,0,1,1,0], [1,1,0,0,0], [0,1,0,0,1], [0,0,0,1,0]]))

    adata = anndata.AnnData(X)
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.var = df 
    adata.obsp['connectivities'] = connectivities

    similarity = DotProduct()
        

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
        
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[-1], 1/(e**(-1) + 1))


    def test_similarityComputer_softmax_masked(self):
        sc = SimilarityComputer()
        self.assertFalse(sc._center_mean, sc._scale_by_norm)

        numerator = np.array([1,2])
        denom = np.array([0,1])
        mask = denom == 0 #[True, False]
        denom[mask] = 1 #[1,1]
        softmax_scale = 1.0
        result, _= sc.softmax_masked(numerator/denom, mask, softmax_scale)

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0)


    def test_call_DotProduct(self): #OK
        sw = SimilarityWrapper.create('dot')
        self.assertFalse(sw._center_mean)
        self.assertFalse(sw._scale_by_norm)

        X = np.array([[1,2,3],[1,2,1],[1,2,1]])
        v = np.array([1,1,1])
        softmax_scale = 0.5

        probs, logits = sw(v, X, softmax_scale)
        self.assertEqual(probs.shape[0], 3)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits[0], 6)
        self.assertEqual(probs[0], 1/(1+2*e**(-1)))


    def test_call_Correlation(self): #OK 
        sw = SimilarityWrapper.create('correlation')
        self.assertTrue(sw._center_mean)
        self.assertTrue(sw._scale_by_norm)

        X = np.array([[1,2,3],[1,2,1],[1,0,0]], dtype=np.float64)
        v = np.array([1,2,1], dtype=np.float64)
        softmax_scale = 0.5

        probs, logits = sw(v, X, softmax_scale)
        self.assertEqual(probs.shape[0], 3)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits[0], 0.)
        self.assertEqual(logits[1], 1.)
        self.assertEqual(logits[2], -0.5)
        self.assertAlmostEqual(probs[0], 0.29175596)
        self.assertAlmostEqual(probs[2], 0.22721977)
        self.assertAlmostEqual(np.sum(probs), 1.0)


    def test_deterministic_inits(self):
        velo = self.velocities.get(self.adata.var['gene']).to_numpy(dtype=np.float64)
        det = Deterministic(self.adata, velo, self.similarity, softmax_scale=1.0)
        self.assertEqual(det._key, 'connectivities')
        self.assertEqual(det._softmax_scale, 1.0)
        self.assertIsInstance(det._similarity, type(self.similarity))
        self.assertEqual(det._velocities.shape, (5,4))
        self.assertEqual(det._adata.shape, (5,4))


    def test_deterministic_uniform(self):
        velo = np.zeros((5,4))
        det = Deterministic(self.adata, velo, self.similarity, softmax_scale=1.0)
        probs, logits = det.uniform(3)
        self.assertEqual(probs.shape[0], 3)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits[0], 0)
        self.assertEqual(probs[0], 1/3)


    def test_deterministic_displacement(self):
        velo = np.zeros((5,4))
        det = Deterministic(self.adata, velo, self.similarity, softmax_scale=1.0)
        neighbors, displacement = det.compute_displacement_vector(0)
        self.assertEqual(displacement.shape, (2,4))
        self.assertEqual(neighbors.shape, (2,))

        expected_result_0 = self.X[1]-self.X[0]
        self.assertEqual(displacement[0][0], expected_result_0[0])
        self.assertEqual(displacement[0][1], expected_result_0[1])
        self.assertEqual(displacement[0][2], expected_result_0[2])
        self.assertEqual(displacement[0][3], expected_result_0[3])


    def test_deterministic_update_transitions_result(self):
        velo = np.zeros((5,4))
        det = Deterministic(self.adata, velo, self.similarity, softmax_scale=1.0)

        probabilities = np.array([0.3, 0.7], dtype= np.float64)
        logits =  np.array([2.,3.], dtype= np.float64)
        neighbors = np.array([1,2])

        det.update_transitions(probabilities=probabilities, logits=logits, neighbors=neighbors)
        self.assertEqual(len(det._logits), 2)
        self.assertEqual(len(det._probabilities), 2)
        self.assertEqual(len(det._indptr), 2)
        self.assertEqual(det._indptr[0], 0)
        self.assertEqual(det._indices[0], 1)
        self.assertEqual(det._probabilities[0], 0.3)

        probabilities = np.array([0.1, 0.3, 0.6], dtype= np.float64)
        logits =  np.array([2. , 3., 9.], dtype= np.float64)
        neighbors = np.array([0, 2 , 3])

        det.update_transitions(probabilities=probabilities, logits=logits, neighbors=neighbors)
        self.assertEqual(det._logits.shape, (5,))
        self.assertEqual(det._probabilities.shape, (5,))
        self.assertEqual(det._indptr.shape, (3,))
        self.assertEqual(det._indptr[-1], 5)
        self.assertEqual(det._indices[-1], 3)
        self.assertEqual(det._probabilities[-1], 0.6)

        probabilities = np.array([0.4, 0.6], dtype= np.float64)
        logits =  np.array([3., 9.], dtype= np.float64)
        neighbors = np.array([0, 1])
        det.update_transitions(probabilities=probabilities, logits=logits, neighbors=neighbors)

        probabilities = np.array([0.7, 0.3], dtype= np.float64)
        logits =  np.array([2. , 3.], dtype= np.float64)
        neighbors = np.array([1,4])
        det.update_transitions(probabilities=probabilities, logits=logits, neighbors=neighbors)

        probabilities = np.array([1.0], dtype= np.float64)
        logits =  np.array([3.], dtype= np.float64)
        neighbors = np.array([3])
        det.update_transitions(probabilities=probabilities, logits=logits, neighbors=neighbors)

        probs, logs = det.result
        self.assertIsInstance(probs, csr_matrix)
        self.assertIsInstance(logs, csr_matrix)
        self.assertTrue(np.isclose(probs.sum(axis=1), 1.0).all())
        #print(probs.sum(axis=1)) ok tutto 1.
        self.assertEqual(probs.shape, (5,5))
        self.assertEqual(logs.shape, (5,5))

    def test_deterministic_call(self):
        velo = np.zeros((5,4))
        det = Deterministic(self.adata, velo, self.similarity, softmax_scale=1.0)

        probs, logs = det() 
        self.assertEqual(probs.shape, (5,5))
        self.assertEqual(logs.shape, (5,5))
        self.assertEqual(probs.toarray()[0][1], 1/2)

        velo = self.velocities.get(self.adata.var['gene']).to_numpy(dtype=np.float64)
        det = Deterministic(self.adata, velo, self.similarity, softmax_scale=1.0)

        probs, logs = det() 
        self.assertEqual(probs.shape, (5,5))
        self.assertEqual(logs.shape, (5,5))


        _, displacement = det.compute_displacement_vector(0)
        v = velo[0]
        p = self.similarity(v, displacement, 1.0)
        self.assertEqual(probs.toarray()[0][1].all(), p[0].all()) 


    
if __name__=='__main__':
    unittest.main()