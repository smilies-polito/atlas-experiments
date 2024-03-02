# SaveSeuratCSV
R [function](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/save_scATAC.R) that enables to save in csv format a Seurat Object for scATAC-seq data (when SeuratDisk doesn't work). \
params: 

    - object: SeuratObject with scATAC-seq data preprocessed using Signac (v5). 
    - path: path of the folder where files should be stored. 
    - assay: Assay for SeuratObject. Default: `'peaks'`.
    - reduction.list: array of dimensionality reductions to store. Default `c('lsi')`.
    - verbose: boolean for verbosity. Default `TRUE`.


output: every file is stored in _path_. 

    - metadata.csv: file storing metadata with header and barcodes.
    - metafeature.csv: file storing metafeature with header and peaks.
    - {reduction}_cell_embdedding.csv: file storing the cell embedding for the reduction method of interest. For example, if {reduction} is LSI, then lsi_cell_embedding.csv contains `Embeddings(object, reduction='lsi')`. 
    - {reduction}_std.csv: file storing the standard deviation for the reduction method of interest. For example, if {reduction} is LSI, then  lsi_cell_std.csv contains `Stdev(object, reduction='lsi')`. 
    - neighbor_idx.csv: file storing the cell neighbor indices, hence `object@neighbors$peaks.nn@nn.idx`
    - neighbor_dist.csv: file storing the cell neighbor distances, hence `object@neighbors$peaks.nn@nn.dist`

# SaveSeuratDATA
R [function](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/save_scATAC.R) that enables to save in csv format the `data` matrix of a SeuratObject for scATAC-seq data.
params: 

    - object: SeuratObject with scATAC-seq data preprocessed using Signac (v5). 
    - path: path of the folder where the file should be stored. 
    - assay: Assay for SeuratObject. Dafult: `'peaks'`.  
    - verbose: boolean for verbosity. Default `TRUE`.


output: file is stored in _path_. 

    - data.csv: file containing sparse reprsentation of the data matrix `summary(object[[assay]]@data)`. The following is a table composed by three columns: "i" is the gene index, "j" is the cell index, "x" is the value-ij of `object[[assay]]@data`. Only non-zero values are stored.


# CSVLoader 
Python [class](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/csvLoader.py) that enables to create an `anndata.AnnData` object from the files saved with [SaveSeuratCSV](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/save_scATAC.R).   \
fields: 

- path: path of the directory with files to be loaded
- _X: csr_matrix of shape n_obs x n_vars
- _metadata: dataframe of shape n_obs x n_features containing values for the observations
- _metafeature: dataframe of shape n_var x n_features containing values for the features
- _obs_names: sequence of shape n_obs containing observations' names
- _var_names: sequence of shape n_var containing variables' names
- embeddings: dictionary with embeddings' names as key and embeddings' representations as values. Representaion is an object of class Reader.  

> `init` 
>> params:
>>> path: path of the folder where files to lead are stored. 

>> output:
>>> a CSVLoader object.

> `set_X`
>> params:
>>> X: matrix for adata.X of size (n_obs x n_var). If None is provided it is created from path/data.csv

>> output:
>>> a CSVLoader object with updated field `._metafeature` and `._var_names`.


> `set_metadata`
>> params:
>>> metadata: dataframe for adata.obs of size (n_obs x n_features). If None is provided it is created from path/metadata.csv.

>> output:
>>> a CSVLoader object with updated field `._metadata` and `._obs_names`.

> `set_metafeature`
>> params:
>>> metafeature: dataframe for adata.var of size (n_var x n_features). If None is provided it is created from path/metafeature.csv.

>> output: 
>>> A CSVLoader object with updated field `._X` (sparse csr_matrix). 

> `set_embedding`
>> params:
>>> embedding: string among 'pca' or 'lsi'  indicating the embedding to load. Default 'pca'. Accepts also an Embedding object. Data are loaded from path/{embedding}_cell_embdedding.csv and path/{embedding}_std.csv.

>> output
>>> a CSVLoader object with updated field `._embeddings`.


> `create_adata`
>> params:
>>> embedding: string or list of strings among 'pca' or 'lsi', indicating the embeddings to load. Default ['pca']. Accepts also an Embedding object or list of Embedding objects. 

>> output: 
>>> a CSVLoader object with updated `._adata` field. This contains an anndata.AnnData object with the following fields: \
    - adata.X  is CSVLoader._X \
    - adata.obs is CSVLoader._metadata and adata.obs_names is CSVLoader._obs_names  \
    - adata.var is CSVLoader._metafeature and adata.var_names is CSVLoader._var_names \
    - adata.uns[{embedding}]['variance'] is CSVLoader._embeddings[{embedding}]._variance if {embedding} is 'lsi' \
    - adata.obsm['X_{embedding}'] is CSVLoader._embeddings[{embedding}]._embedding

# LSIReader
Python [class](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/csvLoader.py) that enables to read LSI embedding files saved using [SaveSeuratCSV](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/save_scATAC.R).   \
fields: 

- path: path of the directory with files to be loaded. It expects to contain lsi_cell_embedding.csv and lsi_std.csv.
- _embedding: matrix of shape n_obs x n_lsi_dimensions with observations embedding from lsi_cell_embedding.csv.
- _variance: array of shape n_lsi_dimensions with variance from lsi_std.csv.

> `init` 
>> params:
>>> path: path of the folder where files to lead are stored. 

>> output:
>>> a LSIReader object.

> `__call__`

>> output:
>>> a LSIReader object with updated field `._embedding` and `._variance`.

# PCAReader
Python [class](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/csvLoader.py) that enables to read PCA embedding files saved using [SaveSeuratCSV](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/save_scATAC.R).  \
Will be removed if not used otherwise fully implemented. 
