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

# ---- Wildcard constraints (prevent greedy matching across delimiters) -------
wildcard_constraints:
    tree     = "[a-z_]+",
    rd       = r"[\d.]+",
    sigma    = r"[\d.]+",
    knn_rna  = r"\d+",
    knn_act  = r"\d+",
    wnn      = r"\d+",

# ---- Targets ----------------------------------------------------------------
PAL_DIR = "output/simulations/palantir"
CR_DIR = "output/simulations/pseudotime_kernel"
PAL_RNA_DIR = "output/simulations/palantir_rna"
CR_DIR_RNA = "output/simulations/pseudotime_kernel_rna"

rule all:
    input:
        expand(
            CR_DIR_RNA + "/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
            tree=TREES, rd=RDS, sigma=SIGMAS,
            knn_rna=KNN_RNAS,
	),
        expand(
            PAL_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            tree=TREES, rd=RDS, sigma=SIGMAS,
            knn_rna=KNN_RNAS, knn_act=KNN_ACTS, wnn=WNNS,
        ),
        expand(
            CR_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            tree=TREES, rd=RDS, sigma=SIGMAS,
            knn_rna=KNN_RNAS, knn_act=KNN_ACTS, wnn=WNNS,
        ),
        expand(
            PAL_RNA_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}.h5ad",
            tree=TREES, rd=RDS, sigma=SIGMAS,
            knn_rna=KNN_RNAS,
        ),

# ---- Per-combination job ----------------------------------------------------
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
        "python3 -m simulated_data.palantir_run "
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
        "python3 -m simulated_data.pseudotime_kernel_rna "
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
