# =============================================================================
# Snakemake pipeline – Palantir simulations over all parameter combinations
#
# Run from the scvemo/ directory:
#   snakemake -n            # dry run
#   snakemake --cores all   # fully parallel (CSV writes are file-locked)
# =============================================================================

# ---- Parameter grid ---------------------------------------------------------
TREES      = [ "three_branches","five_branches"]
RDS        = [0.1, 0.3, 0.5, 0.7, 0.9]
SIGMAS     = [0.1, 0.3, 0.5, 0.7, 0.9]
KNN_RNAS   = [10,30,50,70]
KNN_ACTS   = [10,30,50,70]
WNNS       = [10,30,50,70]

E18_PCS_RNA = [20]
E18_PCS_ACT = [10]
#E18_KNN_RNA = [30, 20, 10, 50, 70]
E18_KNN_RNA = [20]
#E18_KNN_ACT = [30, 20, 10, 50, 70]
E18_KNN_ACT = [50]
#E18_WNN = [30, 20, 10, 50, 70]
E18_WNN = [20]

# ---- Wildcard constraints (prevent greedy matching across delimiters) -------
wildcard_constraints:
    tree     = "[a-z_]+",
    rd       = r"[\d.]+",
    sigma    = r"[\d.]+",
    knn_rna  = r"\d+",
    knn_act  = r"\d+",
    n_pcs_rna  = r"\d+",
    n_pcs_act  = r"\d+",
    wnn      = r"\d+",

# ---- Targets ----------------------------------------------------------------
PAL_DIR = "output/simulations/palantir"
CR_DIR = "output/simulations/pseudotime_kernel"
PAL_RNA_DIR = "output/simulations/palantir_rna"
CR_DIR_RNA = "output/simulations/pseudotime_kernel_rna"
E18_DIR = "output/embryonic_mouse_brain" 
SKIN_DIR = "output/mouse_skin" 

rule all:
    input:
     expand( E18_DIR + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard.h5mu",
        n_pcs_rna = E18_PCS_RNA, n_pcs_act = E18_PCS_ACT, 
        knn_rna = E18_KNN_RNA, knn_act = E18_KNN_ACT,
        wnn = E18_WNN,
    ),
#    expand(
  #          CR_DIR_RNA + "/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
 #           tree=TREES, rd=RDS, sigma=SIGMAS,
 #           knn_rna=KNN_RNAS,
#    ),
#        expand(
 #           PAL_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
#            tree=TREES, rd=RDS, sigma=SIGMAS,
#            knn_rna=KNN_RNAS, knn_act=KNN_ACTS, wnn=WNNS,
#        ),
#        expand(
#            CR_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
#            tree=TREES, rd=RDS, sigma=SIGMAS,
#            knn_rna=KNN_RNAS, knn_act=KNN_ACTS, wnn=WNNS,
#        ),
#        expand(
#            PAL_RNA_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
#            tree=TREES, rd=RDS, sigma=SIGMAS,
#            knn_rna=KNN_RNAS,
#        ),

# ---- Per-combination job ----------------------------------------------------
rule run_preprocessing_skin:
    input:
        atac = "data/mouse_skin/GSM4156597_skin.late.anagen.counts.txt",
        rna = "data/mouse_skin/GSM4156608_skin.late.anagen.rna.counts.txt",
        annotations = "data/mouse_skin/GSM4156597_skin_celltype.txt",
        peaks = "data/mouse_skin/GSM4156597_skin.late.anagen.peaks.bed",
        barcodes = "data/mouse_skin/GSM4156597_skin.late.anagen.barcodes.txt",
        fragment = "data/mouse_skin/GSM4156597_skin.late.anagen.atac.sorted.fragments.bed.gz"
    output:
         features =  SKIN_DIR + "/features.tsv",
         data =  SKIN_DIR + "/skin.h5mu",
    log:
        "logs/mouse_skin/preprocessing.log",
    shell:
        "python3 -m real_data.skin_preprocessing "

rule run_preprocessing_e18:
    input:
        matrix = "data/embryonic_mouse_brain/filtered_feature_bc_matrix",
        annotations = "data/embryonic_mouse_brain/cell_annotations.tsv",
        fragment = "data/embryonic_mouse_brain/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz",
    output:
         features =  E18_DIR + "/features.tsv",
         data =  E18_DIR + "/emb.h5mu",
    log:
        "logs/e18_mouse_brain/preprocessing.log",
    shell:
        "python3 -m real_data.e18_preprocessing "
        "&> {log}"

rule run_e18:
    threads: 3
    input:
        data = "output/embryonic_mouse_brain/emb.h5mu",
        features = "output/embryonic_mouse_brain/features.tsv",
    output:
         hard =  E18_DIR + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard.h5mu",
    log:
        "logs/e18_mouse_brain/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard.log",
    shell:
        "python3 -m real_data.e18_run "
        "--pcs_rna {wildcards.n_pcs_rna} "
        "--pcs_act {wildcards.n_pcs_act} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_activity {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "&> {log}"

rule run_palantir:
    input:
        activity = "data/simulated_data/{tree}/{rd}_{sigma}_activity.tsv",
        spliced  = "data/simulated_data/{tree}/{rd}_{sigma}_spliced.tsv",
        metadata = "data/simulated_data/{tree}/{rd}_{sigma}_metadata.tsv",
    output:
        h5mu_free  = PAL_DIR + "/{tree}_False_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        h5mu_fixed = PAL_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    log:
        "logs/palantir/{tree}_{rd}_{sigma}_{knn_rna}_{knn_act}_{wnn}.log",
    shell:
        "python3 -m simulated_data.new_palantir "
        "--tree {wildcards.tree} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_activity {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "&> {log}"

rule run_cellrank:
    threads: 1
    input:
        activity = "data/simulated_data/{tree}/{rd}_{sigma}_activity.tsv",
        spliced  = "data/simulated_data/{tree}/{rd}_{sigma}_spliced.tsv",
        metadata = "data/simulated_data/{tree}/{rd}_{sigma}_metadata.tsv",
    output:
        h5mu_free  = CR_DIR + "/{tree}_False_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        h5mu_fixed = CR_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    log:
        "logs/pseudotime_kernel/{tree}_{rd}_{sigma}_{knn_rna}_{knn_act}_{wnn}.log",
    shell:
        "python3 -m simulated_data.pseudotime_kernel_run "
        "--tree {wildcards.tree} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_activity {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "&> {log}"

rule run_palantir_rna:
    input:
        spliced  = "data/simulated_data/{tree}/{rd}_{sigma}_spliced.tsv",
        metadata = "data/simulated_data/{tree}/{rd}_{sigma}_metadata.tsv",
    output:
        h5mu_free  = PAL_RNA_DIR + "/{tree}_False_{rd}_{sigma}_{knn_rna}.h5ad",
        h5mu_fixed = PAL_RNA_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
    log:
        "logs/palantir_rna/{tree}_{rd}_{sigma}_{knn_rna}.log",
    shell:
        "python3 -m simulated_data.palantir_rna_run "
        "--tree {wildcards.tree} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--knn_rna {wildcards.knn_rna} "
        "&> {log}"

rule run_cellrank_rna:
    threads: 3
    input:
        spliced  = "data/simulated_data/{tree}/{rd}_{sigma}_spliced.tsv",
        metadata = "data/simulated_data/{tree}/{rd}_{sigma}_metadata.tsv",
    output:
        h5mu_free  = CR_DIR_RNA + "/{tree}_False_{rd}_{sigma}_{knn_rna}.h5ad",
        h5mu_fixed = CR_DIR_RNA + "/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
    log:
        "logs/pseudotime_kernel_rna/{tree}_{rd}_{sigma}_{knn_rna}.log",
    shell:
        "python3 -u -m simulated_data.pseudotime_kernel_rna_run "
        "--tree {wildcards.tree} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--knn_rna {wildcards.knn_rna} "
        "&> {log}"
