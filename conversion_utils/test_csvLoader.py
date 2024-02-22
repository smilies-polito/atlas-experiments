import os
import unittest
import sys
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from csvLoader import *


class Test(unittest.TestCase):

    PATH = os.getcwd()
    
    def test_init_exception(self):
        wrong_path = os.path.join(os.getcwd(), 'fake_folder')
        with self.assertRaises(FileNotFoundError):
            CSVLoader(wrong_path)

    def test_init(self):
        loader = CSVLoader(self.PATH)
        self.assertEqual(self.PATH, loader._path)
        self.assertIsNone(loader._X)
        self.assertIsNone(loader._metadata)
        self.assertIsNone(loader._metafeature)
        self.assertIsNone(loader._adata)
        self.assertFalse(bool(loader._embeddings))

    
    def test_Xparam_numpy(self):
        loader = CSVLoader(self.PATH)
        X = np.array([[1,2,3], [5,6,7], [8,9,10]])
        shape = X.shape
        loader.set_X(X)
        self.assertEqual(shape, loader._X.shape)
        self.assertIsInstance(loader._X, csr_matrix)


    def test_Xparam_csr(self):
        loader = CSVLoader(self.PATH)
        X = np.array([[1,2,3], [5,6,7], [8,9,10]])
        X = csr_matrix(X)
        shape = X.shape
        loader.set_X(X)
        self.assertEqual(shape, loader._X.shape)
        self.assertIsInstance(loader._X, csr_matrix)
    
    
    def test_metafeature_pd(self):
        loader = CSVLoader(self.PATH)
        df = pd.read_csv(os.path.join(self.PATH, 'metafeature.csv'), header=0, index_col=0, sep=',')
        loader.set_metafeature(df)
        self.assertIsNotNone(loader._metafeature)
        self.assertEqual(loader._metafeature.shape[0], 172193)
        self.assertEqual(len(loader._var_names), 172193)


    def test_metafeature_read(self):
        loader = CSVLoader(self.PATH)
        loader.set_metafeature()
        self.assertIsNotNone(loader._metafeature)
        self.assertEqual(loader._metafeature.shape[0], 172193)
        self.assertEqual(len(loader._var_names), 172193)


    def test_metadata_pd(self):
            loader = CSVLoader(self.PATH)
            df = pd.read_csv(os.path.join(self.PATH, 'metadata.csv'), header=0, index_col=0, sep=',')
            loader.set_metadata(df)
            self.assertIsNotNone(loader._metadata)
            self.assertEqual(loader._metadata.shape[0], 4380)
            self.assertEqual(len(loader._obs_names), 4380)


    def test_metadata_pd(self):
            loader = CSVLoader(self.PATH)
            loader.set_metadata()
            self.assertIsNotNone(loader._metadata)
            self.assertEqual(loader._metadata.shape[0], 4380)
            self.assertEqual(len(loader._obs_names), 4380)


    def test_embedding_string(self):
            loader = CSVLoader(self.PATH)
            string = 'lsi'
            loader.set_embedding(string)
            self.assertTrue(bool(loader._embeddings))
            self.assertTrue(loader._embeddings.keys().__contains__(string))
            embedding = loader._embeddings[string]
            self.assertIsInstance(embedding, LSIReader)
            self.assertIsNotNone(embedding._embedding)
            self.assertEqual(embedding._embedding.shape, (4380, 50))


    def test_embedding_embedding(self):
            loader = CSVLoader(self.PATH)
            string = 'lsi'
            loader.set_embedding(Embedding.LSI)
            self.assertTrue(bool(loader._embeddings))
            self.assertTrue(loader._embeddings.keys().__contains__(string))
            embedding = loader._embeddings[string]
            self.assertIsInstance(embedding, LSIReader)
            self.assertIsNotNone(embedding._embedding)
            self.assertEqual(embedding._embedding.shape, (4380, 50))
         

    def test_X_loading(self):
        loader = CSVLoader(self.PATH)
        loader.set_X()
        self.assertIsNotNone(loader._X)
        self.assertEqual(loader._X.shape, (4380,172193))


    def test_adata_single_string(self):
        loader = CSVLoader(self.PATH)       
        string = 'lsi'  
        loader.create_adata(string)
        self.assertIsNotNone(loader._adata)
        adata = loader.adata
        self.assertEqual(adata.shape, (4380,172193))
        self.assertEqual(len(adata.obs_names), 4380)
        self.assertEqual(len(adata.var_names), 172193)
        self.assertIsNotNone(adata.obsm[f'X_{string}'])
        self.assertEqual(adata.obsm[f'X_{string}'].shape, (4380, 50))
        self.assertIsNotNone(adata.uns[string])
        self.assertTrue(adata.uns[string].keys().__contains__('variance'))
        self.assertEqual(len(adata.uns[string]['variance']), 50)


    def test_adata_single_stringList(self):
        loader = CSVLoader(self.PATH)       
        list = ['lsi']  
        string = 'lsi'
        loader.create_adata(list)
        self.assertIsNotNone(loader._adata)
        adata = loader.adata
        self.assertEqual(adata.shape, (4380,172193))
        self.assertEqual(len(adata.obs_names), 4380)
        self.assertEqual(len(adata.var_names), 172193)
        self.assertIsNotNone(adata.obsm[f'X_{string}'])
        self.assertEqual(adata.obsm[f'X_{string}'].shape, (4380, 50))
        self.assertIsNotNone(adata.uns[string])
        self.assertTrue(adata.uns[string].keys().__contains__('variance'))
        self.assertEqual(len(adata.uns[string]['variance']), 50)

    def test_adata_single_embedding(self):
        loader = CSVLoader(self.PATH)       
        string = 'lsi'  
        embdg = Embedding.LSI
        loader.create_adata(embdg)
        self.assertIsNotNone(loader._adata)
        adata = loader.adata
        self.assertEqual(adata.shape, (4380,172193))
        self.assertEqual(len(adata.obs_names), 4380)
        self.assertEqual(len(adata.var_names), 172193)
        self.assertIsNotNone(adata.obsm[f'X_{string}'])
        self.assertEqual(adata.obsm[f'X_{string}'].shape, (4380, 50))
        self.assertIsNotNone(adata.uns[string])
        self.assertTrue(adata.uns[string].keys().__contains__('variance'))
        self.assertEqual(len(adata.uns[string]['variance']), 50)

    def test_adata_single_embeddingList(self):
            loader = CSVLoader(self.PATH)       
            string = 'lsi'  
            embdg =  [Embedding.LSI]
            loader.create_adata(embdg)
            self.assertIsNotNone(loader._adata)
            adata = loader.adata
            self.assertEqual(adata.shape, (4380,172193))
            self.assertEqual(len(adata.obs_names), 4380)
            self.assertEqual(len(adata.var_names), 172193)
            self.assertIsNotNone(adata.obsm[f'X_{string}'])
            self.assertEqual(adata.obsm[f'X_{string}'].shape, (4380, 50))
            self.assertIsNotNone(adata.uns[string])
            self.assertTrue(adata.uns[string].keys().__contains__('variance'))
            self.assertEqual(len(adata.uns[string]['variance']), 50)


if __name__=='__main__':
    Test.PATH = sys.argv.pop()
    unittest.main()