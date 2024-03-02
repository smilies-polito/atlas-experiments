SaveSeuratDATA <- function(object, 
                            path,
                            assay='peaks', 
                            verbose = T){

  if(verbose){print("Save data summary. Might take a while.")}
  data <- summary(object[[assay]]@data)
  write.table(data, file = file.path(path ,"data.csv"), sep=',', row.names = F, col.names = T)
  
}


SaveSeuratCSV <- function(object, 
                          path,
                          assay = 'peaks',
                          reduction.list = c('lsi'),
                          verbose = T){
  
  if(verbose){print("Save metadata")}
  meta.data <- as.data.frame(as.matrix(object@meta.data))
  fwrite(x = meta.data, row.names = T, col.names=T, file = file.path(path,"metadata.csv"))
  
  if(verbose){print("Save metafeature")}
  meta.feature <- as.data.frame(as.matrix(object[[assay]]@meta.features))
  fwrite(x = meta.feature, row.names = T,  col.names=T, file = file.path(path,"metafeature.csv"))
  
  
  for(reduction in reduction.list){
    cell.red.file <- file.path(path, paste0(paste(reduction, "cell", "embedding", sep='_'), '.csv'))
    feature.red.file  <- file.path(path, paste0(paste(reduction, "feature", "embedding", sep='_'), '.csv'))
    std.red.file  <- file.path(path, paste0(paste(reduction, "std", sep='_'), '.csv'))
    cell.embedding <- Embeddings(object, reduction=reduction)
    feature.loadings <- Loadings(object, reduction=reduction)
    std <- Stdev(object, reduction = reduction)
    write.table(cell.embedding, file = cell.red.file, sep=',', row.names = T, col.names = T)
    write.table(feature.loadings, file = feature.red.file, sep=',', row.names = T, col.names = T)
    write.table(std, file = std.red.file, sep=',', row.names = T, col.names = F)
  }
  
  if(verbose){print("Save neighborhood")}
  idx = object@neighbors$peaks.nn@nn.idx
  distances = object@neighbors$peaks.nn@nn.dist
  idx.path <- file.path(path, 'neighbor_idx.csv')
  dist.path <- file.path(path, 'neighbor_dist.csv')
  write.table(idx, file = idx.path, sep=',', row.names = T, col.names = T)
  write.table(distances, file = dist.path, sep=',', row.names = T, col.names = T)  
  

}

