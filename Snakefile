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

MOUSE_HAIR_DATA= DATA_DIR+"/mouse_hair"
MOUSE_HAIR_OUTPUT= OUTPUT_DIR+"/mouse_hair"
MOUSE_HAIR_FOLDER = "/real_data/mouse_hair"

# WILDCARDS
MH_STATES = [4,5,6,7,8]
KNN_RNA = [15]
KNN_ACT = [15]
WNN = ["None"]
N_PCS_RNA = [20]
N_PCS_ACT = [10]

wildcard_constraints:
    states = r"\d+",
    real_states = r"\d+",
    n_pcs_rna = r"\d+",
    n_pcs_act = r"\d+",
    wnn = r"(None|\d+)",
    knn_rna = r"\d+",
    knn_act = r"\d+",


# RULE ALL
rule all:
    input:
    # preprocessing mouse hair dataset
        expand(
        MOUSE_HAIR_OUTPUT + "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        n_pcs_rna = N_PCS_RNA, 
        n_pcs_act= N_PCS_ACT, 
        knn_rna = KNN_RNA, 
        knn_act = KNN_ACT, 
        wnn = WNN,
   ),
    # atlas palantir mouse hair dataset 
        expand(
        MOUSE_HAIR_OUTPUT + "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        n_pcs_rna = N_PCS_RNA, 
        n_pcs_act= N_PCS_ACT, 
        knn_rna = KNN_RNA, 
        knn_act = KNN_ACT, 
        wnn = WNN,
   ),
    # atlas cellrank mouse hair dataset 
        expand(
        MOUSE_HAIR_OUTPUT + "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
        n_pcs_rna = N_PCS_RNA, 
        n_pcs_act= N_PCS_ACT, 
        knn_rna = KNN_RNA, 
        knn_act = KNN_ACT, 
        wnn = WNN,
        n_states = MH_STATES,
   ),
    # rna palantir mouse hair dataset 
        expand(
        MOUSE_HAIR_OUTPUT + "/rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu",
        n_pcs_rna = N_PCS_RNA, 
        n_pcs_act= N_PCS_ACT, 
        knn_rna = KNN_RNA, 
        knn_act = KNN_ACT, 
        wnn = WNN,
   ),
    # rna cellrank mouse hair dataset 
        expand(
        MOUSE_HAIR_OUTPUT + "/rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu",
        n_pcs_rna = N_PCS_RNA, 
        n_pcs_act= N_PCS_ACT, 
        knn_rna = KNN_RNA, 
        knn_act = KNN_ACT, 
        wnn = WNN,
        n_states = MH_STATES,
   ),



rule mouseHair_preprocessing:
    input:
        rna = MOUSE_HAIR_DATA + "/GSM4156608_skin.late.anagen.rna.counts.txt",
        atac = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.counts.txt",
        barcodes = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.barcodes.txt",
        peaks = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.peaks.bed",
        annotations = MOUSE_HAIR_DATA + "/GSM4156597_skin_celltype.txt",
        fragment_file = MOUSE_HAIR_DATA + "/GSM4156597_skin.late.anagen.atac.fragments.sorted.bed.gz",
    output:
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu"
    ), 
    log:
        ( LOG_DIR + "/_mouse_hair_preprocessing_" + 
        "{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log"
    ), 
    shell:
        "python3 -m real_data.mouse_hair.preprocessing "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"

rule mouseHair_atlas_palantir:
    input:
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu"
    ), 
    output:
        pseudotime = ( MOUSE_HAIR_OUTPUT + 
        "/pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv"
    ),
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu"
    ), 
    log:
        ( LOG_DIR + "/_mouse_hair_palantir_" + 
        "{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log"
    ), 
    shell:
        "python3 -m real_data.mouse_hair.figure1_hf_atlas_palantir "
        "--n_jobs {threads} "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"

rule mouseHair_atlas_cellrank:
    input:
        pseudotime = ( MOUSE_HAIR_OUTPUT + 
        "/pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv"
    ), 
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu"
    ), 
    output:
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu"
    ), 
    log:
        ( LOG_DIR + "/_mouse_hair_cellrank_" + 
        "{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log"
    ), 
    shell:
        "python3 -m real_data.mouse_hair.figure1_hf_atlas_cellrank "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} &> {log}"

rule mouseHair_rna_palantir:
    input:
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.h5mu"
    ), 
    output:
        tsv = ( MOUSE_HAIR_OUTPUT + 
        "/rna_pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.tsv"
    ), 
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/rna_palantir_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu"
    ), 
    log:
        ( LOG_DIR + "/_mouse_hair_rna_palantir_" + 
        "{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.log"
    ), 
    shell:
        "python3 -m real_data.mouse_hair.supplementary5_mh_rna_palantir "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} &> {log}"

rule mouseHair_rna_cellrank:
    input:
        pseudotime = ( MOUSE_HAIR_OUTPUT + 
        "/rna_pseudotime_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}.tsv"
    ), 
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu"
    ), 
    output:
        h5mu = ( MOUSE_HAIR_OUTPUT + 
        "/rna_cellrank_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.h5mu"
    ), 
    log:
        ( LOG_DIR + "/_mouse_hair_rna_cellrank_" + 
        "{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_{n_states}.log"
    ), 
    shell:
        "python3 -m real_data.mouse_hair.supplementary5_mh_rna_palantir "
        "--knn_rna {wildcards.knn_rna} "
        "--knn_act {wildcards.knn_act} "
        "--wnn {wildcards.wnn} "
        "--n_pcs_rna {wildcards.n_pcs_rna} "
        "--n_pcs_act {wildcards.n_pcs_act} "
        "--n_states {wildcards.n_states} &> {log}"

