import anndata
import unittest
import numpy as np 
import pandas as pd

from math import e
from scipy.sparse import csr_matrix
from transitionMatrix import Similarity, SimilarityComputer, SimilarityWrapper, Cosine, Correlation, DotProduct
from transitionMatrix import Deterministic, TransitionMatrix

np.random.seed(52)

class Test(unittest.TestCase):
    obs_names = ['cell1', 'cell2', 'cell3', 'cell4', 'cell5']
    var_names = ['peak1', 'peak2', 'peak3', 'peak4']
    X = np.random.uniform(size=(5,4))
    X = csr_matrix(X)
    gene_names = ['gene0', 'gene1','gene2', 'gene3', 'gene4', 'gene5']
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

        expected_result_0 = (self.X[1].A-self.X[0].A).flatten()
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


    def test_transitionMatrix_init_error(self):
        velo1 = np.zeros((5,5))
        with self.assertRaises(ValueError) as ve:
            tm = TransitionMatrix(self.adata, velo1)
        self.assertEqual(str(ve.exception),"Shapes of velocity array (5, 5) and adata (5, 4) do not match.")

        velo2 = np.zeros((6,4))
        with self.assertRaises(ValueError) as ve:
            tm = TransitionMatrix(self.adata, velo2)
        self.assertEqual(str(ve.exception),"Shapes of velocity array (6, 4) and adata (5, 4) do not match.")

        with self.assertRaises(ValueError) as ve:
            TransitionMatrix(self.adata, self.velocities)
        self.assertEqual(str(ve.exception), "With pd.DataFrame promoter_key field is mandatory" )

        velocity_sized = self.velocities[1:]
        with self.assertRaises(ValueError) as ve:
            TransitionMatrix(self.adata, velocity_sized, promoter_key='gene')
        self.assertEqual(str(ve.exception), "Barcordes no not match." )

        new_barcodes = np.array(['cell10', 'cell2', 'cell3', 'cell4', 'cell5'])
        velocity_wrong = self.velocities.copy().set_index(new_barcodes)
        with self.assertRaises(ValueError) as ve:
            TransitionMatrix(self.adata, velocity_wrong, promoter_key='gene')
        self.assertEqual(str(ve.exception), "Barcordes no not match." )    

        velocity_genes = self.velocities[self.gene_names[1:]]
        with self.assertRaises(ValueError) as ve:
            TransitionMatrix(self.adata, velocity_genes, promoter_key='gene')
        self.assertEqual(str(ve.exception), "Not all genes in adata.var[gene] are in velocities")    

        with self.assertRaises(ValueError) as ve:
            TransitionMatrix(self.adata, self.velocities, promoter_key='ciao')
        self.assertEqual(str(ve.exception), "ciao not in adata.var")    


    
    def test_transitionMatrix_init_numpy(self):
        velo1 = np.zeros((5,4))
        velo1[0][0] = np.nan 
        tm = TransitionMatrix(self.adata, velocities=velo1)
        self.assertIsNotNone(tm._velocities)
        self.assertIsInstance(tm._velocities, np.ndarray)
        self.assertEqual(tm._velocities[0][0], 0.) #nans have been replaced


    def test_transitionMatrix_init_df(self):
        nan_velocities = self.velocities.copy()
        nan_velocities['gene5'] = np.ones(nan_velocities.shape[0]) * np.nan
        tm = TransitionMatrix(self.adata, nan_velocities, promoter_key='gene', softmax_scale=0.5)
        self.assertIsNotNone(tm._velocities)
        self.assertIsInstance(tm._velocities, np.ndarray)
        self.assertEqual(tm._velocities[0][0], 0.) #nans have been replaced
        self.assertEqual(tm._velocities.shape, (5,4))
        self.assertIsNotNone(tm._adata)
        self.assertEqual(tm._adata.shape,(5,4))
        self.assertEqual(tm._softmax_scale, 0.5)
        self.assertEqual(tm._promoter_key, 'gene')
        self.assertEqual(tm.key, 'gene')


    def test_softmax_scale_computation(self):
        tm =  TransitionMatrix(self.adata, self.velocities, promoter_key='gene')
        self.assertIsNone(tm._softmax_scale)
        res = tm.estimate_softmax_scale(similarity=DotProduct(), key='connectivities')
        
        det = Deterministic(self.adata, 
                            self.velocities.get(self.adata.var['gene']).to_numpy(dtype=np.float64),
                            similarity=DotProduct(),
                            softmax_scale=1.0,
                            key = 'connectivities')
        _, logits = det()
        self.assertEqual(res, 1.0/np.median(np.abs(logits.data)))


    def test_compute_transition_matrix_str(self):
        tm =  TransitionMatrix(self.adata, self.velocities, promoter_key='gene')
        tm.compute_transition_matrix('dot')
        self.assertIsNotNone(tm._softmax_scale)
        self.assertIsNotNone(tm._logits)
        self.assertIsNotNone(tm._transition_matrix)


    def test_compute_transition_matrix_similarityCom(self):
        tm =  TransitionMatrix(self.adata, self.velocities, promoter_key='gene')
        tm.compute_transition_matrix(Cosine())
        self.assertIsNotNone(tm._softmax_scale)
        self.assertIsNotNone(tm._logits)
        self.assertIsNotNone(tm._transition_matrix)


    def test_compute_transition_matrix_similarity(self):
        tm =  TransitionMatrix(self.adata, self.velocities, promoter_key='gene')
        tm.compute_transition_matrix(Similarity.DOT_PRODUCT)
        self.assertIsNotNone(tm._softmax_scale)
        self.assertIsNotNone(tm._logits)
        self.assertIsNotNone(tm._transition_matrix)



if __name__=='__main__':
    unittest.main()