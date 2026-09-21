# =============================================================================
# Snakemake pipeline – trajectory inference over all parameter combinations
#
# Threading model
# ---------------
# Two families of rules need OPPOSITE threading strategies:
#
#   * PETSc/SLEPc rules  (cellrank, pseudotime_kernel): the heavy step is a
#     linear/eigen solve inside PETSc/SLEPc, which scales with BLAS/OpenMP
#     THREADS. joblib fan-out here is nearly useless (few macrostates) and
#     causes oversubscription. -> BLAS = {threads}, n_jobs = 1.
#
#   * joblib rules  (palantir): fate probabilities are parallelised over many
#     worker PROCESSES via joblib/loky. Each worker must run single-threaded
#     BLAS or the workers fight over cores. -> BLAS = 1, n_jobs = {threads}.
#
#   * light rules  (preprocessing, expression): mostly I/O + scanpy; BLAS = 1
#     is fine.
#
# Threading env vars are injected PER RULE (not globally in run.sh), so each
# job on the cluster gets the profile it actually needs.
#
# Run locally:
#   snakemake -n                       # dry run
#   snakemake --cores 16               # single node
# Run on SLURM (one sbatch per rule, each with its own cpus/mem):
#   snakemake --executor slurm --jobs 20 ...
# =============================================================================

OUTPUT_DIR = "output"
DATA_DIR = "data"
LOG_DIR = "logs"

MOUSE_HAIR_DATA = DATA_DIR + "/mouse_hair"
MOUSE_HAIR_OUTPUT = OUTPUT_DIR + "/mouse_hair"
MOUSE_HAIR_FOLDER = "/real_data/mouse_hair"
HUMAN_BRAIN_DATA = DATA_DIR + "/human_brain"
HUMAN_BRAIN_OUTPUT = OUTPUT_DIR + "/human_brain"
HUMAN_BRAIN_FOLDER = "/real_data/human_brain"
E18_DATA = DATA_DIR + "/embryonic_mouse_brain"
E18_OUTPUT = OUTPUT_DIR + "/embryonic_mouse_brain"
E18_FOLDER = "/real_data/embryonic_mouse_brain"
SIMULATION_MUDATA = OUTPUT_DIR + "/simulations" + "/mudata"
SIMULATION_DATA = DATA_DIR + "/simulated_data"
SIMULATION_OUTPUT = OUTPUT_DIR + "/simulations"

# WILDCARDS
MH_STATES = [4, 5, 6, 7, 8]
HB_STATES = [4, 5, 6, 7, 8]
E18_STATES = [6, 12]
REMOVED = {1: "removed", 0: "non_removed"}
INVERSE_REMOVED = {v: k for k, v in REMOVED.items()}
KNN_RNA = [15]
KNN_ACT = [15]
WNN = ["None"]
N_PCS_RNA = [20]
N_PCS_ACT = [10]

TREES = ["five_branches", "three_branches"]
RDS = [0.1, 0.3, 0.5, 0.7, 0.9]
SIGMAS = [0.1, 0.3, 0.5, 0.7, 0.9]
SIM_KNN_RNAS = [10, 30, 50, 70]
SIM_KNN_ACTS = [10, 30, 50, 70]
SIM_WNNS = [10, 30, 50, 70]

# -----------------------------------------------------------------------------
# Threading configuration
# -----------------------------------------------------------------------------
# Thread counts per rule family. Tune these to your node size on serics_cpu.
PETSC_THREADS = 8      # cellrank / pseudotime_kernel: BLAS/PETSc threads
PALANTIR_THREADS = 16  # palantir: joblib worker processes
LIGHT_THREADS = 4      # preprocessing / expression


def env_threads(n):
    return (
        "export OMP_NUM_THREADS={n}; "
        "export OPENBLAS_NUM_THREADS={n}; "
        "export MKL_NUM_THREADS={n}; "
        "export NUMEXPR_NUM_THREADS={n}; "
        "export VECLIB_MAXIMUM_THREADS={n}; "
    ).format(n=n)


# BLAS=1 for the whole process; joblib gets the cores via --n_jobs {threads}.
ENV_JOBLIB = env_threads(1)
# BLAS/PETSc get all the cores; --n_jobs 1 disables joblib fan-out.
# NB: the export uses the shell var $SNK_THREADS set from {threads} at runtime.
ENV_PETSC = (
    "export OMP_NUM_THREADS={threads}; "
    "export OPENBLAS_NUM_THREADS={threads}; "
    "export MKL_NUM_THREADS={threads}; "
    "export NUMEXPR_NUM_THREADS={threads}; "
    "export VECLIB_MAXIMUM_THREADS={threads}; "
)


wildcard_constraints:
    states = r"\d+",
    real_states = r"\d+",
    n_pcs_rna = r"\d+",
    n_pcs_act = r"\d+",
    wnn = r"(None|\d+)",
    knn_rna = r"\d+",
    knn_act = r"\d+",
    rd = r"0\.\d+",
    sigma = r"0\.\d+",
    tree = r"three_branches|five_branches",


# RULE ALL
rule all:
    input:
        # preprocessing mouse hair dataset
        expand(
            MOUSE_HAIR_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # atlas palantir mouse hair dataset
        expand(
            MOUSE_HAIR_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # atlas cellrank mouse hair dataset
        expand(
            MOUSE_HAIR_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            n_states=MH_STATES,
        ),
        # rna palantir mouse hair dataset
        expand(
            MOUSE_HAIR_OUTPUT + "/rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # rna cellrank mouse hair dataset
        expand(
            MOUSE_HAIR_OUTPUT + "/rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            n_states=MH_STATES,
        ),
        # preprocessing e18 mouse brain dataset
        expand(
            E18_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # atlas palantir e18 mouse brain dataset
        expand(
            E18_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # atlas cellrank e18 mouse brain dataset
        expand(
            E18_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            n_states=E18_STATES,
        ),
        # rna palantir e18 mouse brain dataset
        expand(
            E18_OUTPUT + "/rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # rna cellrank e18 mouse brain dataset
        expand(
            E18_OUTPUT + "/rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            n_states=E18_STATES,
        ),
        # e18 mouse brain expression plots
        expand(
            E18_OUTPUT + "/expression_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # preprocessing human brain dataset
        expand(
            HUMAN_BRAIN_OUTPUT + "/{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            removed=list(REMOVED.values()),
        ),
        # atlas palantir human brain dataset
        expand(
            HUMAN_BRAIN_OUTPUT + "/palantir_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            removed=list(REMOVED.values()),
        ),
        # atlas cellrank human brain dataset
        expand(
            HUMAN_BRAIN_OUTPUT + "/cellrank_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            n_states=HB_STATES,
            removed=list(REMOVED.values()),
        ),
        # rna palantir human brain dataset
        expand(
            HUMAN_BRAIN_OUTPUT + "/rna_palantir_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            removed=list(REMOVED.values()),
        ),
        # rna cellrank human brain dataset
        expand(
            HUMAN_BRAIN_OUTPUT + "/rna_cellrank_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
            n_states=HB_STATES,
            removed=list(REMOVED.values()),
        ),
        # human brain expression plots
        expand(
            HUMAN_BRAIN_OUTPUT + "/expression_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done",
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=KNN_RNA,
            knn_act=KNN_ACT,
            wnn=WNN,
        ),
        # preprocess synthetic
        expand(
            SIMULATION_MUDATA + "/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
            tree=TREES,
            rd=RDS,
            sigma=SIGMAS,
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=SIM_KNN_RNAS,
            knn_act=SIM_KNN_ACTS,
            wnn=SIM_WNNS,
        ),
        # atlas palantir synthetic
        expand(
            SIMULATION_OUTPUT + "/palantir/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done",
            tree=TREES,
            rd=RDS,
            sigma=SIGMAS,
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=SIM_KNN_RNAS,
            knn_act=SIM_KNN_ACTS,
            wnn=SIM_WNNS,
        ),
        # atlas cellrank synthetic
        expand(
            SIMULATION_OUTPUT + "/pseudotime_kernel/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done",
            tree=TREES,
            rd=RDS,
            sigma=SIGMAS,
            n_pcs_rna=N_PCS_RNA,
            n_pcs_act=N_PCS_ACT,
            knn_rna=SIM_KNN_RNAS,
            knn_act=SIM_KNN_ACTS,
            wnn=SIM_WNNS,
        ),
        # rna palantir synthetic
        expand(
            SIMULATION_OUTPUT + "/palantir_rna/{tree}_{rd}:{sigma}_{n_pcs_rna}_{knn_rna}.done",
            tree=TREES,
            rd=RDS,
            sigma=SIGMAS,
            n_pcs_rna=N_PCS_RNA,
            knn_rna=SIM_KNN_RNAS,
        ),
        # atlas cellrank rna synthetic
        expand(
            SIMULATION_OUTPUT + "/pseudotime_kernel_rna/{tree}_{rd}:{sigma}_{n_pcs_rna}_{knn_rna}.done",
            tree=TREES,
            rd=RDS,
            sigma=SIGMAS,
            n_pcs_rna=N_PCS_RNA,
            knn_rna=SIM_KNN_RNAS,
        ),


# =============================================================================
# MOUSE HAIR
# =============================================================================

rule mouseHair_preprocessing:
    input:
        rna = MOUSE_HAIR_DATA + "/GSM4156608_skin.late.anagen.rna.counts.txt",
        atac = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.counts.txt",
        barcodes = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.barcodes.txt",
        peaks = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.peaks.bed",
        annotations = MOUSE_HAIR_DATA + "/GSM4156597_skin_celltype.txt",
        fragment_file = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.atac.fragments.sorted.bed.gz",
    output:
        h5mu = MOUSE_HAIR_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: LIGHT_THREADS
    log:
        LOG_DIR + "/_mouse_hair_preprocessing_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.mouse_hair.preprocessing "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule mouseHair_atlas_palantir:
    input:
        h5mu = MOUSE_HAIR_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        pseudotime = MOUSE_HAIR_OUTPUT + "/pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = MOUSE_HAIR_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_mouse_hair_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.mouse_hair.figure1_hf_atlas_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule mouseHair_atlas_cellrank:
    input:
        pseudotime = MOUSE_HAIR_OUTPUT + "/pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = MOUSE_HAIR_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        h5mu = MOUSE_HAIR_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_mouse_hair_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log",
    shell:
        ENV_PETSC +
        "python3 -m real_data.mouse_hair.figure1_hf_atlas_cellrank "
        "--n_jobs 1 "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} &> {log}"


rule mouseHair_rna_palantir:
    input:
        h5mu = MOUSE_HAIR_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        tsv = MOUSE_HAIR_OUTPUT + "/rna_pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = MOUSE_HAIR_OUTPUT + "/rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_mouse_hair_rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.mouse_hair.supplementary5_mh_rna_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule mouseHair_rna_cellrank:
    input:
        pseudotime = MOUSE_HAIR_OUTPUT + "/rna_pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = MOUSE_HAIR_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    output:
        h5mu = MOUSE_HAIR_OUTPUT + "/rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_mouse_hair_rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log",
    shell:
        ENV_PETSC +
        "python3 -m real_data.mouse_hair.supplementary5_mh_rna_cellrank "
        "--n_jobs 1 "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} &> {log}"
# =============================================================================
# E18 MOUSE BRAIN
# =============================================================================

rule e18brain_preprocessing:
    input:
	matrix = E18_DATA + "/filtered_feature_bc_matrix",
	ann1 = E18_DATA + "/cell_annotations.tsv",
	ann2 = E18_DATA + "/archvelo_labels.csv",
	umap = E18_DATA + "/multivelo_umap.csv",
	features = E18_DATA + "/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz",

    output:
        h5mu = E18_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: LIGHT_THREADS
    log:
        LOG_DIR + "/_e18_brain_preprocessing_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.embryonic_mouse_brain.preprocessing "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule e18brain_atlas_palantir:
    input:
        h5mu = E18_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        pseudotime = E18_OUTPUT + "/pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = E18_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_embryonic_brain_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.embryonic_mouse_brain.supplementary6_e18_atlas_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule e18brain_atlas_cellrank:
    input:
        pseudotime = E18_OUTPUT + "/pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = E18_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        h5mu = E18_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_embryonic_brain_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log",
    shell:
        ENV_PETSC + 
        "python3 -m real_data.embryonic_mouse_brain.supplementary6_e18_atlas_cellrank "
        "--n_jobs 1 "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} &> {log}"


rule e18brain_rna_palantir:
    input:
        h5mu = E18_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        tsv = E18_OUTPUT + "/rna_pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = E18_OUTPUT + "/rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_embryonic_brain_rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.embryonic_mouse_brain.supplementary7_e18_rna_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule e18brain_rna_cellrank:
    input:
        pseudotime = E18_OUTPUT + "/rna_pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = E18_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    output:
        h5mu = E18_OUTPUT + "/rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_embryonic_brain_rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log",
    shell:
        ENV_PETSC +
        "python3 -m real_data.embryonic_mouse_brain.supplementary7_e18_rna_cellrank "
        "--n_jobs 1 "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} &> {log}"

rule e18brain_expression:
    input:
        h5mu = E18_OUTPUT + "/non_removed_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        touch(E18_OUTPUT + "/expression_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done"),
    threads: LIGHT_THREADS
    log:
        LOG_DIR + "/e18_brain_expression_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.embryonic_mouse_brain.supplementary8_e18_gex "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"

# =============================================================================
# HUMAN BRAIN
# =============================================================================

rule humanBrain_preprocessing:
    input:
        clusters = HUMAN_BRAIN_DATA + "/GSE162170_multiome_cluster_names.txt",
        metadata = HUMAN_BRAIN_DATA + "/GSE162170_multiome_cell_metadata.txt",
        rna = HUMAN_BRAIN_DATA + "/GSE162170_multiome_rna_counts.tsv.gz",
        activity = HUMAN_BRAIN_DATA + "/GSE162170_multiome_atac_gene_activities.tsv.gz",
        outlier = HUMAN_BRAIN_DATA + "/to_remove.tsv",
    output:
        h5mu = HUMAN_BRAIN_OUTPUT + "/{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    params:
        removed = lambda wildcards: INVERSE_REMOVED[wildcards.removed],
    threads: LIGHT_THREADS
    log:
        LOG_DIR + "/human_brain_preprocessing_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.human_brain.preprocessing "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--removed {params.removed} &> {log}"


rule humanBrain_expression:
    input:
        h5mu = HUMAN_BRAIN_OUTPUT + "/non_removed_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        touch(HUMAN_BRAIN_OUTPUT + "/expression_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done"),
    threads: LIGHT_THREADS
    log:
        LOG_DIR + "/human_brain_expression_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.human_brain.supplementary9_hb_gex "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule humanBrain_atlas_palantir:
    input:
        h5mu = HUMAN_BRAIN_OUTPUT + "/{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        h5mu = HUMAN_BRAIN_OUTPUT + "/palantir_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        tsv = HUMAN_BRAIN_OUTPUT + "/pseudotime_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
    params:
        removed = lambda wildcards: INVERSE_REMOVED[wildcards.removed],
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_human_brain_atlas_palantir_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.human_brain.supplementary1012_hb_atlas_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--removed {params.removed} &> {log}"


rule humanBrain_atlas_cellrank:
    input:
        h5mu = HUMAN_BRAIN_OUTPUT + "/{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        tsv = HUMAN_BRAIN_OUTPUT + "/pseudotime_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
    output:
        h5mu = HUMAN_BRAIN_OUTPUT + "/cellrank_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    params:
        removed = lambda wildcards: INVERSE_REMOVED[wildcards.removed],
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_human_brain_atlas_cellrank_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log",
    shell:
        ENV_PETSC +
        "python3 -m real_data.human_brain.supplementary1012_hb_atlas_cellrank "
        "--n_jobs 1 "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} "
        "--removed {params.removed} &> {log}"


rule humanBrain_rna_palantir:
    input:
        h5mu = HUMAN_BRAIN_OUTPUT + "/{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        tsv = HUMAN_BRAIN_OUTPUT + "/rna_pseudotime_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
        h5mu = HUMAN_BRAIN_OUTPUT + "/rna_palantir_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    params:
        removed = lambda wildcards: INVERSE_REMOVED[wildcards.removed],
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_human_brain_rna_palantir{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m real_data.human_brain.supplementary1113_hb_rna_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--removed {params.removed} &> {log}"


rule humanBrain_rna_cellrank:
    input:
        h5mu = HUMAN_BRAIN_OUTPUT + "/{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        tsv = HUMAN_BRAIN_OUTPUT + "/rna_pseudotime_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv",
    output:
        h5mu = HUMAN_BRAIN_OUTPUT + "/rna_cellrank_{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
    params:
        removed = lambda wildcards: INVERSE_REMOVED[wildcards.removed],
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_human_brain_rna_cellrank{removed}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log",
    shell:
        ENV_PETSC +
        "python3 -m real_data.human_brain.supplementary1113_hb_rna_cellrank "
        "--n_jobs 1 "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} "
        "--removed {params.removed} &> {log}"


# =============================================================================
# SYNTHETIC
# =============================================================================

rule preprocess_synthetic:
    input:
        activity = SIMULATION_DATA + "/{rd}_{sigma}_activity.tsv",
        metadata = SIMULATION_DATA + "/{rd}_{sigma}_metadata.tsv",
        spliced = SIMULATION_DATA + "/{rd}_{sigma}_spliced.tsv",
    output:
        h5mu = SIMULATION_MUDATA + "/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    threads: LIGHT_THREADS
    log:
        LOG_DIR + "/_simulation_preprocessing_{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m simulated_data.preprocessing "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--tree {wildcards.tree} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule synthetic_palantir:
    input:
        h5mu = SIMULATION_MUDATA + "/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        sentinel = touch(SIMULATION_OUTPUT + "/palantir/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done"),
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_simulated_palantir_{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_JOBLIB +
        "python3 -m simulated_data.palantir_run "
        "--n_jobs {threads} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--tree {wildcards.tree} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule synthetic_pseudotime_kernel:
    input:
        h5mu = SIMULATION_MUDATA + "/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
    output:
        sentinel = touch(SIMULATION_OUTPUT + "/pseudotime_kernel/{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.done"),
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_simulated_cellrank_{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log",
    shell:
        ENV_PETSC +
        "python3 -m simulated_data.pseudotime_kernel_run "
        "--n_jobs 1 "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--tree {wildcards.tree} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"


rule synthetic_palantir_rna:
    input:
        h5mu = SIMULATION_MUDATA + "/{tree}_{rd}:{sigma}_{n_pcs_rna}:10_{knn_rna}:30:30.h5mu",
    output:
        sentinel = touch(SIMULATION_OUTPUT + "/palantir_rna/{tree}_{rd}:{sigma}_{n_pcs_rna}_{knn_rna}.done"),
    threads: PALANTIR_THREADS
    log:
        LOG_DIR + "/_simulated_palantir_rna_{tree}_{rd}:{sigma}_{n_pcs_rna}_{knn_rna}:.log",
    shell:
        ENV_JOBLIB +
        "python3 -m simulated_data.palantir_rna_run "
        "--n_jobs {threads} "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--tree {wildcards.tree} "
        "--knn_rna {wildcards.knn_rna} "
        "--n_pcs_rna {wildcards.n_pcs_rna} &> {log}"


rule synthetic_pseudotime_kernel_rna:
    input:
        h5mu = SIMULATION_MUDATA + "/{tree}_{rd}:{sigma}_{n_pcs_rna}:10_{knn_rna}:30:30.h5mu",
    output:
        sentinel = touch(SIMULATION_OUTPUT + "/pseudotime_kernel_rna/{tree}_{rd}:{sigma}_{n_pcs_rna}_{knn_rna}.done"),
    threads: PETSC_THREADS
    log:
        LOG_DIR + "/_simulated_cellrank_rna_{tree}_{rd}:{sigma}_{n_pcs_rna}_{knn_rna}:.log",
    shell:
        ENV_PETSC +
        "python3 -m simulated_data.pseudotime_kernel_rna_run "
        "--n_jobs 1 "
        "--rd {wildcards.rd} "
        "--sigma {wildcards.sigma} "
        "--tree {wildcards.tree} "
        "--knn_rna {wildcards.knn_rna} "
        "--n_pcs_rna {wildcards.n_pcs_rna} &> {log}"
