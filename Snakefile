# =============================================================================
# Snakemake pipeline – Palantir simulations over all parameter combinations
#
# Run from the scvemo/ directory:
#   snakemake -n            # dry run
#   snakemake --cores all   # fully parallel (CSV writes are file-locked)
# =============================================================================

OUTPUT_DIR= "output"
DATA_DIR= "data"
LOG_DIR= "logs"

SIM_OUTPUT= OUTPUT_DIR+"/simulations"
SIM_DATA= DATA_DIR+"/simulated_data"
HUMAN_BRAIN_DATA= DATA_DIR+"/human_brain"
HUMAN_BRAIN_OUTPUT= OUTPUT_DIR+"/human_brain"
E18_BRAIN_OUTPUT= OUTPUT_DIR+"/embryonic_mouse_brain"
E18_BRAIN_DATA= DATA_DIR+"/embryonic_mouse_brain"
MOUSE_HAIR_DATA= DATA_DIR+"/mouse_hair"
MOUSE_HAIR_OUTPUT= OUTPUT_DIR+"/mouse_hair"

# WILDCARDS
HB_REMOVED = {"brain": 0, "brainR": 1}
HB_STATES = [5, 10]
E18_STATES = [6, 12]
MH_STATES = [4, 8]

REAL_KNN_RNA = [10, 30, 50, 70]
REAL_KNN_ACT = [10, 30, 50, 70]
REAL_WNN = [10, 30, 50, 70]
REAL_STATES = [4,5,6,7,8,9,10,11,12]
ALGORITHM = ["pseudotime_kernel", "palantir"]

TREES = ["three_branches", "five_branches"]
RDS = [0.1, 0.3, 0.5, 0.7, 0.9]
SIGMAS = [0.1, 0.3, 0.5, 0.7, 0.9]
KNN_RNAS = [10,30,50,70]
KNN_ACTS = [10,30,50,70]
WNNS = [10,30,50,70]


wildcard_constraints:
	prefix = "brain|brainR",
	states = r"\d+",
	real_states = r"\d+",
	real_rna = r"\d+",
	real_act = r"\d+",
	real_wnn = r"\d+",
	knn_rna = r"\d+",
	knn_act = r"\d+",
	wnn = r"\d+",
	rd = r"0\.\d+",
	sigma = r"0\.\d+",
	tree = r"three_branches|five_branches",
	algorithm = r"palantir|pseudotime_kernel",


# RULE ALL
rule all:
	input:
		expand( HUMAN_BRAIN_OUTPUT + "/{prefix}.h5mu", 
			prefix = HB_REMOVED.keys(),
		),
		E18_BRAIN_OUTPUT + "/features.tsv",
		E18_BRAIN_OUTPUT + "/emb.h5mu",
		MOUSE_HAIR_OUTPUT + "/features.tsv",
		MOUSE_HAIR_OUTPUT + "/hair.h5mu",
		expand( E18_BRAIN_OUTPUT + "/20:10_15:15:None_hard_{states}states.h5mu",
			states = E18_STATES,
		), 
		expand( E18_BRAIN_OUTPUT + "/rna_{states}.h5mu",	
			states = E18_STATES,
		),
		E18_BRAIN_OUTPUT + "/.supplementary_8.done",
		expand( MOUSE_HAIR_OUTPUT + "/20:10_15:15:None_hard_{states}states.h5mu",
			states = MH_STATES,
		), 
		expand( MOUSE_HAIR_OUTPUT + "/rna_{states}.h5mu",	
			states = MH_STATES,
		),
		expand( HUMAN_BRAIN_OUTPUT + "/{prefix}_{states}.h5mu",
			states = HB_STATES,
			prefix = HB_REMOVED.keys(),
		), 
		expand( HUMAN_BRAIN_OUTPUT + "/rna_{prefix}_{states}.h5mu",
			states = HB_STATES,
			prefix = HB_REMOVED.keys(),
		), 
		HUMAN_BRAIN_OUTPUT + "/.supplementary_9.done",
		MOUSE_HAIR_OUTPUT + "/.lineages.done",
		expand( MOUSE_HAIR_OUTPUT + "/hyper_done/rna{real_rna}_act{real_act}_wnn{real_wnn}_states{real_states}.done",
			real_rna = REAL_KNN_RNA,
			real_act = REAL_KNN_ACT,
			real_wnn = REAL_WNN,
			real_states = REAL_STATES,
		),
#		expand( SIM_OUTPUT + "/ATLAS_{algorithm}_performance.done",
#			algorithm = ALGORITHM
#		),
#		expand( SIM_OUTPUT + "/{algorithm}_benchmark.done",
#			algorithm = ALGORITHM
#		),
		expand( SIM_OUTPUT + "/palantir/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
			tree = TREES,
			rd = RDS,
			sigma = SIGMAS,
			knn_rna = KNN_RNAS,
			knn_act = KNN_ACTS,
			wnn = WNNS,
		),
		expand( SIM_OUTPUT + "/pseudotime_kernel/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
			tree = TREES,
			rd = RDS,
			sigma = SIGMAS,
			knn_rna = KNN_RNAS,
			knn_act = KNN_ACTS,
			wnn = WNNS,
		),
		expand( SIM_OUTPUT + "/palantir_rna/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
			tree = TREES,
			rd = RDS,
			sigma = SIGMAS,
			knn_rna = KNN_RNAS,
		),
		expand( SIM_OUTPUT + "/pseudotime_kernel_rna/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
			tree = TREES,
			rd = RDS,
			sigma = SIGMAS,
			knn_rna = KNN_RNAS,
		),


rule humanBrain_preprocessing:
	input:
		clusters = HUMAN_BRAIN_DATA + "/GSE162170_multiome_cluster_names.txt",
		metadata = HUMAN_BRAIN_DATA + "/GSE162170_multiome_cell_metadata.txt",
		rna = HUMAN_BRAIN_DATA + "/GSE162170_multiome_rna_counts.tsv.gz",
		activity = HUMAN_BRAIN_DATA + "/GSE162170_multiome_atac_gene_activities.tsv.gz",
		outlier = HUMAN_BRAIN_DATA + "/to_remove.tsv",
	output:
		h5mu = HUMAN_BRAIN_OUTPUT + "/{prefix}.h5mu",
	params: 
		removed = lambda wildcards: HB_REMOVED[wildcards.prefix],
	log: 
		LOG_DIR + "/HB_preprocessing_{prefix}.log",
	shell:
		"python3 -m real_data.brain_preprocessing --removed {params.removed} &> {log}"

rule e18brain_preprocessing:
	input:
		filtered_feature_bc = E18_BRAIN_DATA + "/filtered_feature_bc_matrix",
		annotations = E18_BRAIN_DATA + "/cell_annotations.tsv",
		fragment_file = E18_BRAIN_DATA + "/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz",
	output:
		feature_path = E18_BRAIN_OUTPUT + "/features.tsv",
		h5mu = E18_BRAIN_OUTPUT + "/emb.h5mu",
	log:
		LOG_DIR + "/E18_preprocessing.log",
	shell:
		"python3 -m real_data.e18_preprocessing &> {log}"

rule mouseHair_preprocessing:
	input:
		rna = MOUSE_HAIR_DATA + "/GSM4156608_skin.late.anagen.rna.counts.txt",
		atac = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.counts.txt",
		barcodes = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.barcodes.txt",
		peaks = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.peaks.bed",
		annotations = MOUSE_HAIR_DATA + "/GSM4156597_skin_celltype.txt",
		fragment_file = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.atac.fragments.sorted.bed.gz",
	output:
		feature_path = MOUSE_HAIR_OUTPUT + "/features.tsv",
		h5mu = MOUSE_HAIR_OUTPUT + "/hair.h5mu",
	log:
		LOG_DIR + "/MH_preprocessing.log",
	shell:
		"python3 -m real_data.mouse_hair_preprocessing &> {log}"

rule supp6_e18brain_atlas:
	input: 
		data = E18_BRAIN_OUTPUT + "/emb.h5mu",
		feature_path = E18_BRAIN_OUTPUT + "/features.tsv",
	output:
		res = E18_BRAIN_OUTPUT + "/20:10_15:15:None_hard_{states}states.h5mu",
	log:
		LOG_DIR + "/E18_atlas_{states}states.log",
	shell: 
		"python3 -m real_data.supplementary6_e18_atlas --n_states {wildcards.states} &> {log}"

rule supp7_e18brain_rna:
	input:
		data = E18_BRAIN_OUTPUT + "/20:10_15:15:None_hard_{states}states.h5mu",
	output:
		h5mu = E18_BRAIN_OUTPUT + "/rna_{states}.h5mu",	
	log:
		LOG_DIR + "/E18_rna_{states}states.log",
	shell:
		"python3 -m real_data.supplementary7_e18_rna --n_states {wildcards.states} &> {log}"

rule supp8_e18brain_gex:
	input: 
		data = E18_BRAIN_OUTPUT + "/20:10_15:15:None_hard_6states.h5mu",
	output:
		sentinel = touch(E18_BRAIN_OUTPUT + "/.supplementary_8.done"),
	log:
		LOG_DIR + "/E18_genes.log",
	shell:
		"python3 -m real_data.supplementary8_e18_gex >& {log}"
	
rule fig1_mouseHair_atlas:
	input: 
		data = MOUSE_HAIR_OUTPUT + "/hair.h5mu",
	output:
		res = MOUSE_HAIR_OUTPUT + "/20:10_15:15:None_hard_{states}states.h5mu",
	log:
		LOG_DIR + "/MH_atlas_{states}states.log",
	shell: 
		"python3 -m real_data.figure1_hf_atlas --n_states {wildcards.states} &> {log}"

rule supp5_mouseHair_rna:
	input:
		data = MOUSE_HAIR_OUTPUT + "/20:10_15:15:None_hard_{states}states.h5mu",
	output:
		h5mu = MOUSE_HAIR_OUTPUT + "/rna_{states}.h5mu",	
	log:
		LOG_DIR + "/MH_rna_{states}states.log",
	shell:
		"python3 -m real_data.supplementary5_mh_rna --n_states {wildcards.states} &> {log}"

rule supp1012_humanBrain_atlas:
	input:
		data = HUMAN_BRAIN_OUTPUT + "/{prefix}.h5mu",
	output: 
		h5mu = HUMAN_BRAIN_OUTPUT + "/{prefix}_{states}.h5mu",
	log:
		LOG_DIR + "/HB_atlas_{prefix}_{states}.log",
	params: 
		removed = lambda wildcards: HB_REMOVED[wildcards.prefix],
	shell: 
		"python3 -m real_data.supplementary1012_hb_atlas --n_states {wildcards.states} --removed {params.removed} &> {log}"

rule supp1113_humanBrain_rna:
	input:
		data = HUMAN_BRAIN_OUTPUT + "/{prefix}_{states}.h5mu",
	output: 
		h5mu = HUMAN_BRAIN_OUTPUT + "/rna_{prefix}_{states}.h5mu",
	log:
		LOG_DIR + "/HB_rna_{prefix}_{states}.log",
	params: 
		removed = lambda wildcards: HB_REMOVED[wildcards.prefix],
	shell: 
		"python3 -m real_data.supplementary1113_hb_rna --n_states {wildcards.states} --removed {params.removed} &> {log}"

rule supp9_humanBrain_gex:
	input:
		data = HUMAN_BRAIN_OUTPUT + "/brain.h5mu",
	output:
		sentinel = touch(HUMAN_BRAIN_OUTPUT + "/.supplementary_9.done"),
	log:
		LOG_DIR + "/HM_genes.log",
	shell:
		"python3 -m real_data.supplementary9_hb_gex >& {log}"

rule lineages:
	input:
		data = MOUSE_HAIR_OUTPUT + "/20:10_15:15:None_hard_8states.h5mu",
	output: 
		sentinel = touch(MOUSE_HAIR_OUTPUT + "/.lineages.done"),
	params:
		strategy = "pseudotime-kernel",
	log:
		LOG_DIR + "/HB_lineages.log",
	shell:
		"python3 -m real_data.figure1_hf_lineages --strategy {params.strategy} --h5mu {input.data} &> {log}"

rule hyper_parameters:
	input:
		data = MOUSE_HAIR_OUTPUT + "/hair.h5mu",
	output:
		sentinel = touch(MOUSE_HAIR_OUTPUT+ "/hyper_done/rna{real_rna}_act{real_act}_wnn{real_wnn}_states{real_states}.done")
	log:
		LOG_DIR + "/hyper_done/rna{real_rna}_act{real_act}_wnn{real_wnn}_states{real_states}.log"
	shell:
		"python3 -m real_data.hyperparams "
		"--rna {wildcards.real_rna} "
		"--act {wildcards.real_act} "
		"--wnn {wildcards.real_wnn} "
		"--states {wildcards.real_states} "
		"&> {log}"
	
#rule supplementary_3_4:
#	input:
#		results = SIM_OUTPUT + "/{algorithm}/results.csv",
#	output:
#		sentinel = touch(SIM_OUTPUT + "/ATLAS_{algorithm}_performance.done"),
#	log: 
#		LOG_DIR + "/ATLAS_{algorithm}_performance.log",
#	shell:
#		"python3 -m supplementary.supplementary_figures_3_4 --algorithm {wildcards.algorithm} &> {log}"
	
		
#rule table_3_4:
#	input:
#		results = SIM_OUTPUT + "/{algorithm}/results.csv",
#		results = SIM_OUTPUT + "/{algorithm}_rna/results.csv",
#	output:
#		sentinel = touch(SIM_OUTPUT + "/{algorithm}_benchmark.done"),
#	log: 
#		LOG_DIR + "/{algorithm}_benchmark.log",
#	shell:
#		"python3 -m supplementary.supplementary_tables_3_4 --algorithm {wildcards.algorithm} &> {log}"
	
rule synthetic_palantir:
	input:
		activity = SIM_DATA + "/{tree}/{rd}_{sigma}_activity.tsv",
		spliced = SIM_DATA + "/{tree}/{rd}_{sigma}_spliced.tsv",
		metadata = SIM_DATA + "/{tree}/{rd}_{sigma}_metadata.tsv",  
	output:
		h5mu_free = SIM_OUTPUT + "/palantir/{tree}_False_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
		h5mu_fixed = SIM_OUTPUT + "/palantir/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
	log:
		LOG_DIR + "/palantir_{tree}_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.log",
	shell: 
		"python3 -m simulated_data.palantir_run "
		"--tree {wildcards.tree} "	
		"--rd {wildcards.rd} "	
		"--sigma {wildcards.sigma} "	
		"--knn_rna {wildcards.knn_rna} "	
		"--knn_activity {wildcards.knn_act} "	
		"--wnn {wildcards.wnn} &> {log} "	

rule synthetic_pseudotime_kernel:
	threads: 3
	input:
		activity = SIM_DATA + "/{tree}/{rd}_{sigma}_activity.tsv",
		spliced = SIM_DATA + "/{tree}/{rd}_{sigma}_spliced.tsv", 
		metadata = SIM_DATA + "/{tree}/{rd}_{sigma}_metadata.tsv",  
	output:
		h5mu_free = SIM_OUTPUT + "/pseudotime_kernel/{tree}_False_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
		h5mu_fixed = SIM_OUTPUT + "/pseudotime_kernel/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
	log:
		LOG_DIR + "/pseudotime_kernel_{tree}_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.log",
	shell: 
		"python3 -m simulated_data.pseudotime_kernel_run "
		"--tree {wildcards.tree} "	
		"--rd {wildcards.rd} "	
		"--sigma {wildcards.sigma} "	
		"--knn_rna {wildcards.knn_rna} "	
		"--knn_activity {wildcards.knn_act} "	
		"--wnn {wildcards.wnn} &> {log} "	

rule synthetic_pseudotime_kernel_rna:
	threads: 3
	input:
		spliced = SIM_DATA + "/{tree}/{rd}_{sigma}_spliced.tsv",
		metadata = SIM_DATA + "/{tree}/{rd}_{sigma}_metadata.tsv",  
	output:
		h5mu_free = SIM_OUTPUT + "/pseudotime_kernel_rna/{tree}_False_{rd}_{sigma}_{knn_rna}.h5ad",
		h5mu_fixed = SIM_OUTPUT + "/pseudotime_kernel_rna/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
	log:
		LOG_DIR + "/pseudotime_kernel_{tree}_{rd}_{sigma}_{knn_rna}.log",
	shell: 
		"python3 -m simulated_data.pseudotime_kernel_rna_run "
		"--tree {wildcards.tree} "	
		"--rd {wildcards.rd} "	
		"--sigma {wildcards.sigma} "	
		"--knn_rna {wildcards.knn_rna} &> {log} "	

rule synthetic_palantir_rna:
	input:
		spliced = SIM_DATA + "/{tree}/{rd}_{sigma}_spliced.tsv", 
		metadata = SIM_DATA + "/{tree}/{rd}_{sigma}_metadata.tsv",  
	output:
		h5mu_free = SIM_OUTPUT + "/palantir_rna/{tree}_False_{rd}_{sigma}_{knn_rna}.h5ad",
		h5mu_fixed = SIM_OUTPUT + "/palantir_rna/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
	log:
		LOG_DIR + "/palantir_{tree}_{rd}_{sigma}_{knn_rna}.log",
	shell: 
		"python3 -m simulated_data.palantir_rna_run "
		"--tree {wildcards.tree} "	
		"--rd {wildcards.rd} "	
		"--sigma {wildcards.sigma} "	
		"--knn_rna {wildcards.knn_rna} &> {log} "	





