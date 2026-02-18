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
SIM_DIR = "data/simulated_data/simulations/palantir"

rule all:
    input:
        expand(
            SIM_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            tree=TREES, rd=RDS, sigma=SIGMAS,
            knn_rna=KNN_RNAS, knn_act=KNN_ACTS, wnn=WNNS,
        ),

# ---- Per-combination job ----------------------------------------------------
rule run_palantir:
    input:
        activity = "data/simulated_data/{tree}/{rd}_{sigma}_activity.tsv",
        spliced  = "data/simulated_data/{tree}/{rd}_{sigma}_spliced.tsv",
        metadata = "data/simulated_data/{tree}/{rd}_{sigma}_metadata.tsv",
    output:
        h5mu_free  = SIM_DIR + "/{tree}_False_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        h5mu_fixed = SIM_DIR + "/{tree}_True_{rd}_{sigma}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    log:
        "logs/palantir/{tree}_{rd}_{sigma}_{knn_rna}_{knn_act}_{wnn}.log",
    shell:
        "python3 -m simulated_data.palantir_pipeline "
        "--tree {wildcards.tree} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_activity {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "&> {log}"
