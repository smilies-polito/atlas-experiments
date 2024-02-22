# SaveSeuratCSV
R [function](https://gitlabtsgroup.polito.it/root/scvemo/-/blob/developer/conversion_utils/save_scATAC.R) that enables to save in csv format a Seurat Object for scATAC-seq data (when SeuratDisk doesn't work). \
params: \
    - adata: SeuratObject with scATAC-seq data preprocessed using Signac. \
    - _path_: path of the folder where files are stored. It should either be an empty folder, otherwise a sub-folder named "SeuratCSV" is created. \
    - assay: Assay for SeuratObject. Default: 'peaks'.\
    - reduction.list: array of dimensionality reductions to store. Default c('lsi', 'umap').\
    - verbose: boolean for verbosity. Default TRUE.



output: Every file is stored in _path_. \
    - _metadata.csv_ : file storing metadata with header and barcodes.\
    - _metafeature.csv_ : file storing metafeature with header and peaks.\
    - _\{reduction\}\_cell\_embdedding.csv_ : file storing the cell embedding for the _reduction_. For example, if reduction is LSI, then _lsi\_cell\_embedding.csv_ contains `Embeddings(adata, reduction='lsi')`. \
    -  _\{reduction\}\_cell\_std.csv_ : file storing the standard deviation for the _reduction_. For example, is reduction is LSI, then  _lsi\_cell\_std.csv_ contains `Stdev(adata, reduction='lsi')`.\ 
    - _data.csv_ : file containing sparse reprsentation of the data matrix `summary(adata[[assay]]@data)`. The following is a table composed by three columns: "i" is the gene index, "j" is the cell index, "x" is the value-ij of `adata[[assay]]@data`. Only non-zero values are stored. 


# csvLoader.py
Python class that creates an anndata.AnnData object with values from csv files. 
ADD DESCRIPTION AND FORMATS
