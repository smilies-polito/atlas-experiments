# Save Seurat Object as a series of CSV files
# object the Seurat Object to save
# path is the folder path in which to save the csv
# reduction.list is the name of dimension reductions to save
# verbose boolean indicating whether to display messages on console

SaveSeuratCSV <- function(object, 
                          path,
                          assay = 'peaks',
                          reduction.list = c('lsi'),
                          verbose = T){
  
  if(!dir.exists(path)){
    print("Directory does not exist. Creating")
    dir.create(path)
  }
  
  if(length(list.files(path))>0){
    print("Directory not empty. Creating sub-folder SeuratCSV")
    path = file.path(path, 'SeuratCSV')
    dir.create(path)
  }
  
  
  if(verbose){print("Save metadata")}
  meta.data <- as.data.frame(as.matrix(object@meta.data))
  fwrite(x = meta.data, row.names = T, col.names=T, file = file.path(path,"metadata.csv"))
  
  if(verbose){print("Save metafeature")}
  meta.feature <- as.data.frame(as.matrix(object[[assay]]@meta.features))
  fwrite(x = meta.feature, row.names = T,  col.names=T, file = file.path(path,"metafeature.csv"))
  
  
  for(reduction in reduction.list){
    if(verbose){print(paste("Save reduction:", reduction))}
    cell.red.file <- file.path(path, paste0(paste(reduction, "cell", "embedding", sep='_'), '.csv'))
    feature.red.file  <- file.path(path, paste0(paste(reduction, "feature", "embedding", sep='_'), '.csv'))
    std.red.file  <- file.path(path, paste0(paste(reduction, "std", sep='_'), '.csv'))
    cell.embedding <- Embeddings(object, reduction=reduction)
    feature.loadings <- Loadings(object, reduction=reduction)
    std <- Stdev(adata, reduction = reduction)
    write.table(cell.embedding, file = cell.red.file, sep=',', row.names = T, col.names = T)
    write.table(feature.loadings, file = feature.red.file, sep=',', row.names = T, col.names = T)
    write.table(std, file = std.red.file, sep=',', row.names = T, col.names = F)
  }
  
  if(verbose){print("Save data summary. Might take a while.")}
  data <- summary(object[[assay]]@data)
  write.table(data, file = file.path(path ,"data.csv"), sep=',', row.names = F, col.names = T)
}


path <- file.path(file.path(getwd(), 'signac_data'), 'E18_embryonic_mouse_brain_ATAC')
SaveSeuratCSV(adata, path)
