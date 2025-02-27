import os, gc
import muon as mu
import scanpy as sc
import pandas as pd
import anndata as ad

def create_feature_map(rna, strand:bool = False):
	features = rna.var.copy()
	columns = ["Chromosome", "Start", "End"]
	if strand:
		columns.append("Strand")
	features = features[columns]
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 
	return features

data_path = ... 
fragment_path = os.path.join(os.getcwd(), "fragment")


rna = sc.read_h5ad( ... )
atac = rna[:, rna.var.modality=="Peaks"]
rna = rna[:, rna.var.modality=="Gene Expression"]
gc.collect() #removing old anndata 

gtf = pd.read_csv( ... , sep="\t", index_col=0)
gtf.set_index("gene_name", inplace=True)
gtf.rename(columns={"start":"Start", "end":"End", "seqname":"Chromosome", "strand":"Strand"}, inplace=True)

rna.var = rna.var.merge(gtf, how="left", left_index=True, right_index=True)
rna = rna[:, ~rna.var["Chromosome"].isna()] #removing non protein coding genes
rna = rna[:, rna.var["Chromosome"]!="chrM"] #removing mithocondrial chromosome

# RNA preprocessing
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, n_top_genes=3000, subset=True)

features = create_feature_map(rna, strand=True)
anatomical_sites = rna.obs.anatomical_site.unique()
run_ids = rna.obs.run_id.unique()
gc.collect() 

for site in anatomical_sites:
	subRna = rna[rna.obs.anatomical_site==site]
	subAtac = atac[atac.obs.anatomical_site==site]
	activity = None
	for run in run_ids:
		if run in subAtac.obs.run_id.unique():
			runAtac = subAtac[subAtac.obs.run_id==run].copy()
			runAtac.obs_names = runAtac.obs_names.map(lambda name: name.split("-")[1] + "-1") # required to match fragments file names
			mu.atac.tl.locate_fragments(runAtac, fragments=os.path.join(fragment_path, f"{run}_atac_fragments.tsv.gz"))
			subAct = mu.atac.tl.count_fragments_features(runAtac, features= features, stranded=True)
			subAct.obs_names = run + "-" + subAct.obs_names.map(lambda name: name.split("-")[0]) #required then to match barcodes with RNA again 
			if activity is None:
				activity = subAct
			else:
				activity = ad.concat([activity, subAct], axis="obs", index_unique=None)
	subRna.write_h5ad(os.path.join(os.getcwd(), f"{site}_RNA.h5ad"))
	activity.write_h5ad(os.path.join(os.getcwd(), f"{site}_ACTIVITY.h5ad"))
	gc.collect()




