import os
import anndata
import numpy as np
import pandas as pd

from enum import Enum 
from scipy.sparse import csr_matrix, coo_matrix
from typing import Literal, Optional, Union, Sequence 

def shrink_lsi_dimensions(adata, n_dims: int, remove_first: bool = True):
    """
    Function that selects the LSI dimensions to keep for umap and neighbor computation. 
    params:
        n_dims: integer describing the number of dimensions to keep. Default 50.
        remove_first: boolesn describing if to remove the first LSI_dimension. Usually it is highly correlated to sequencing depth 
        and most time it is not considered in Signac for neighbor and umap computations. Default True.
    """
    try: 
        X = adata.obsm['X_lsi']
        if n_dims > X.shape[1]:
            print('Specified dimensions exceed matrix dimensionality. Keeping present dimensions.')
        else:
            X = X[:, :n_dims]
        
        if remove_first:
            X = X[:, 1:]

        adata.obsm['X_lsi'] = X
    except:
        print('Impossible to select LSI dimensions.')



class Embedding(Enum):
    """Enum for the embeddings"""
    LSI = 'lsi'
    PCA = 'pca'
    UMAP = 'umap'


class EmbeddingWrapper:
    """Class that allows to create and manage different embeddings"""  
    def create(embedding, path):
        """Returns a reader object given a specific embedding"""
        return LSIReader(path) if embedding == Embedding.LSI else PCAReader(path) if embedding == Embedding.PCA else UMAPReader(path)


class Reader: 
    """General reader class"""
    def __init__(self, path):
        self._path = path
        self._embedding = None
    
    def __call__(self):
        pass

    @property
    def embedding(self):
        return self._embedding


class LSIReader(Reader):
    """Class that reads LSI data"""
    def __init__(self, path):
        super().__init__(path)
        self._variance = None

    def __call__(self):
        """Reads LSI embedding for cells and variance of every LSI dimension"""
        file_name = os.path.join(self._path, 'lsi_cell_embedding.csv')
        lsi = pd.read_csv(file_name, sep=',', header=0, index_col=0)
        self._embedding = lsi.to_numpy()

        file_name = os.path.join(self._path, 'lsi_std.csv')
        stdev = pd.read_csv(file_name, index_col=0, header=None, sep=',')
        self._variance = stdev.to_numpy().flatten() ** 2

    @property
    def variance(self):
        return self._variance


class PCAReader(Reader): 
    """Class that reads PCA data"""
    def __init__(self, path):
        super().__init__(path)
    
    def __call__(self):
        pass


class UMAPReader(Reader):
    """Class that reads UMAP data"""
    def __init__(self, path):
        super().__init__(path)

    def __call__(self):
        file_name = os.path.join(self._path, 'umap_cell_embedding.csv')
        self._embedding = pd.read_csv(file_name, index_col =0, header=0, sep=',').values    

    
class CSVLoader:
    """
    Class that reads data from Signac preprocessing
    """
    def __init__(self, path: str):
        """
        path is the directory where the signac files are stored
        """
        if not os.path.exists(path):
            raise FileNotFoundError("Not a directory.")
        
        self._path = path
        self._X = None
        self._metadata = None
        self._metafeature = None
        self._embeddings = {}
        self._adata = None
        self._neighbors = None
                
 
    def set_X(self, 
              X : Optional[Union[np.array, csr_matrix]] = None
        ):
        """
        Loads data. Either provided with the parameters or read by the data.csv file.  
        Becomes adata.X

        params:
            X: scipy.sparse.csr_matrix or numpy.array of size (n_obs, n_var). Default is None
        """
        if isinstance(X, np.ndarray):
            X = csr_matrix(X)

        elif X is None:
            try: 
                file_path = os.path.join(self._path, 'data.csv')
                df = pd.read_csv(file_path, header=0, sep=',')
                df.i -=1 #R indices start from 1
                df.j -=1 #R indices start from 1
                X = coo_matrix((df.x, (df.j, df.i)))
                X = X.tocsr()
            except: 
                print("The data file does not exist. Cannot load data.")
        
        self._X = X


    def set_metadata(self, 
                    metadata: Optional[pd.DataFrame] = None
        ):
        """
        Loads metadata. Either provided with the parameters or read by the metadata.csv file. 
        Becomes adata.obs

        params:
            metadata: a (n_obs, n_features) pandas dataframe with index_col being obs names. Default None
        """
        if metadata is None:
            try:
                file_path = os.path.join(self._path, 'metadata.csv')    
                metadata= pd.read_csv(file_path, header=0, index_col=0, sep=',')
            except:
                print("The metadata file does not exist. Cannot load metadata.")

        self._metadata = metadata
        self._obs_names = metadata.index


    def set_metafeature(self, 
                        metafeature: Optional[pd.DataFrame] = None
        ):
        """
        Loads metafeature. Either provided with the parameters or read by the metafeature.csv file. 
        Becomes adata.var

        params:
            metafeature: a (n_vars, n_features) pandas dataframe with index_col being var names. Default None
        """
        if metafeature is None:
            try:        
                file_path = os.path.join(self._path, 'metafeature.csv')  
                metafeature = pd.read_csv(file_path, header=0, index_col=0, sep=',')
            except:
                print("The metafeature file does not exists. Cannot load metafeature.")
        
        self._metafeature = metafeature
        self._var_names = metafeature.index


    def set_embedding(self, 
                      embedding: Union[Literal['lsi', 'pca', 'umap'], Embedding] = Embedding.PCA
        ):
        """
        Loads an embedding. 

        params:
            embedding: a string in ['lsi', 'pca'] or an object of class Embedding. Default embedding is PCA.  
        """      
        if isinstance(embedding, str):
            embedding = Embedding(embedding)

        reader = EmbeddingWrapper.create(embedding, self._path)
        reader()
        self._embeddings[embedding.value] = reader


    def set_neighbors(self):
        """
        Function that loads neighborhood. 
        """    
        try:     
            idx_path = os.path.join(self._path, 'neighbor_idx.csv')  
            dist_path = os.path.join(self._path, 'neighbor_dist.csv')
            indices = pd.read_csv(idx_path, header=0, index_col=0, sep=',').values
            n_obs = indices.shape[0]

            indptr = np.apply_along_axis(lambda r: len(r), 1, indices)
            indptr = np.insert(indptr, [0], [0])

            indices  -= 1 #R indices start from 1
            indices = indices.flatten()

            distances = pd.read_csv(idx_path, header=0, index_col=0, sep=',').values.flatten()

            self._neighbors = csr_matrix((distances, indices, indptr), shape=(n_obs, n_obs))
        except:
            print("Unable to load neighborhood.")
        
    
    def create_adata(self, 
                     embeddings : Optional[Union[Sequence[str], Literal['lsi', 'pca', 'umap'], Embedding, Sequence[Embedding]]] = ['pca'],
                     neighbors: bool = False
        ):
        """
        Function that creates adata from csv
        params: 
            embeddings: list of embeddings one wants to read for the data either string, list of strings, object of cass Embedding,
            list of objects of class Embedding. Default is 'pca' 
            neighbors: boolean indicated whether to load the neighborhood. Default is False. 
        """

        if(self._X is None):
            print('Loading data, metadata and metafeature')
            self.set_X()
            self.set_metadata()
            self.set_metafeature()

            print('Loading embeddings')
            if isinstance(embeddings, str) or isinstance(embeddings, Embedding):
                embeddings = [embeddings]

            for emdb in embeddings:
                self.set_embedding(emdb)

        adata = anndata.AnnData(X = self._X, obs = self._metadata, var = self._metafeature)
        adata.obs_names = self._obs_names
        adata.var_names = self._var_names

        for embeddingKey, reader in self._embeddings.items():
            adata.obsm[f'X_{embeddingKey}'] = reader.embedding
            if embeddingKey == 'lsi': #adding variance for lsi 
                adata.uns[f'{embeddingKey}'] = {'variance': reader.variance} 

        if neighbors:
            self.set_neighbors()
            adata.obsp['distances'] = self._neighbors
        
        self._adata = adata

    @property
    def adata(self):
        return self._adata
        

            
        

