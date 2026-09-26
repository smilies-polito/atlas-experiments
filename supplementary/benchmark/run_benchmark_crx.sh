#!/usr/bin/env bash

set -euo pipefail

SIF="${SIF:?set SIF=/path/to/image.sif}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/output/benchmark}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/output/benchmark/results_crx}"
PSEUDOTIME_DIR="${PSEUDOTIME_DIR:-$DATA_DIR/pseudotime}"
CORES="${CORES:-0-15}"               # physical cores of ONE NUMA node (see `lscpu`)
N_REPS="${N_REPS:-5}"
STEPS="${STEPS:-compute_kernel run pipeline}"
RUN_TIMEOUT="${RUN_TIMEOUT:-6h}"     # per single run; a hung run is recorded, not fatal
MAX_LOAD="${MAX_LOAD:-}"             # e.g. 4: wait before each run until the 1-min load is below it
ORDER_SEED="${ORDER_SEED:-2026}"
DATASET="${DATASET:-BMMC}"
N_STATES="${N_STATES:-10}"           # fixed number of GPCCA macrostates
N_COMPONENTS="${N_COMPONENTS:-20}"   # Schur vectors
USE_PETSC="${USE_PETSC:-1}"
ALLOW_OVERLAP="${ALLOW_OVERLAP:-1}"  # initial and terminal states may share cells
CLUSTER_KEY="${CLUSTER_KEY:-cell_type}"
EXTRA_BINDS="${EXTRA_BINDS:-}"       # comma-separated, if data or code live elsewhere
PYTHON="${PYTHON:-}"                 # python inside the image; auto-detected if empty
BLAS_THREADS="${BLAS_THREADS:-1}"    # implicit BLAS/OpenMP threads; n_jobs is the only parallelism knob

RUNTIME="$(command -v apptainer || command -v singularity || true)"
[[ -n "$RUNTIME" ]] || { echo "neither apptainer nor singularity found" >&2; exit 1; }
[[ -f "$SIF" ]] || { echo "image not found: $SIF" >&2; exit 1; }
SIF="$(realpath "$SIF")"; PROJECT_ROOT="$(realpath "$PROJECT_ROOT")"
DATA_DIR="$(realpath "$DATA_DIR")"; mkdir -p "$OUT_DIR"; OUT_DIR="$(realpath "$OUT_DIR")"
[[ -d "$PSEUDOTIME_DIR" ]] || { echo "pseudotime folder not found: $PSEUDOTIME_DIR (run prepare_crx_pseudotime.py first)" >&2; exit 1; }
PSEUDOTIME_DIR="$(realpath "$PSEUDOTIME_DIR")"

N_JOBS="${N_JOBS:-$(taskset -c "$CORES" nproc)}"
(( N_JOBS <= $(taskset -c "$CORES" nproc) )) || { echo "N_JOBS > cores in CORES=$CORES" >&2; exit 1; }

BINDS="$PROJECT_ROOT"
for d in "$DATA_DIR" "$OUT_DIR" "$PSEUDOTIME_DIR"; do
    [[ "$d" == "$PROJECT_ROOT"* ]] || BINDS="$BINDS,$d"
done
[[ -z "$EXTRA_BINDS" ]] || BINDS="$BINDS,$EXTRA_BINDS"

pass_env() {
    local name="$1" value="$2"
    export "SINGULARITYENV_${name}=${value}" "APPTAINERENV_${name}=${value}"
}
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
         NUMBA_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
    pass_env "$v" "$BLAS_THREADS"
done
pass_env PYTHONHASHSEED 0
pass_env PYTHONUNBUFFERED 1
pass_env PYTHONPATH "$PROJECT_ROOT"
pass_env BENCH_CONTAINER "$(basename "$SIF")"

ctr() {
    taskset -c "$CORES" "$RUNTIME" exec --cleanenv --bind "$BINDS" --pwd "$PROJECT_ROOT" "$SIF" "$@"
}

if [[ -z "$PYTHON" ]]; then
    PYTHON="$(ctr sh -c 'for p in python python3 /opt/conda/bin/python /opt/conda/envs/*/bin/python; do
                  command -v "$p" 2>/dev/null && break; done' | head -n 1 || true)"
fi
[[ -n "$PYTHON" ]] || { echo "no python found in the image: set PYTHON=/path/inside/image" >&2; exit 1; }
echo "python in container: $PYTHON"

RAW_DIR="$OUT_DIR/raw"; LOG_DIR="$OUT_DIR/logs"
mkdir -p "$RAW_DIR" "$LOG_DIR"

exec 9>"$OUT_DIR/.lock"
flock -n 9 || { echo "another benchmark is already running on $OUT_DIR" >&2; exit 1; }
if pgrep -u "$(id -u)" -f "^(bash|/bin/bash|/usr/bin/bash) .*run_benchmark_pex\.sh" >/dev/null \
   || pgrep -u "$(id -u)" -f " -m supplementary\.benchmark\.(benchmark_pex|prepare_crx_pseudotime)" >/dev/null; then
    echo "the Palantir benchmark or the pseudotime preparation is running: wait for it to finish" >&2; exit 1
fi

FAILURES="$OUT_DIR/failures.tsv"
[[ -f "$FAILURES" ]] || printf "data_file\tstep\trep\texit_code\n" > "$FAILURES"

echo "computing image checksum..."
pass_env BENCH_CONTAINER_SHA256 "$(sha256sum "$SIF" | cut -d' ' -f1)"

ctr "$PYTHON" -c "import atlas, cellrank, muon, psutil, threadpoolctl" \
    || { echo "environment check inside the container failed" >&2; exit 1; }
if (( USE_PETSC )); then
    ctr "$PYTHON" -c "import petsc4py, slepc4py" \
        || { echo "USE_PETSC=1 but petsc4py/slepc4py are not importable in the container" >&2; exit 1; }
fi

shopt -s nullglob
DATA_FILES=("$DATA_DIR"/data_*.h5mu)
(( ${#DATA_FILES[@]} > 0 )) || { echo "no data_*.h5mu in $DATA_DIR" >&2; exit 1; }
for f in "${DATA_FILES[@]}"; do
    pt="$PSEUDOTIME_DIR/pseudotime_$(basename "$f" .h5mu).tsv"
    [[ -s "$pt" ]] || { echo "missing pseudotime for ${f##*/}: $pt" >&2; exit 1; }
done

echo "host=$(hostname)  runtime=$(basename "$RUNTIME")  image=$(basename "$SIF")"
echo "cores=$CORES  n_jobs=$N_JOBS  blas_threads=$BLAS_THREADS  reps=$N_REPS"
echo "datasets: ${DATA_FILES[*]##*/}"
echo "steps: $STEPS"
echo "n_states=$N_STATES  n_components=$N_COMPONENTS  use_petsc=$USE_PETSC"

JOBLIST="$OUT_DIR/joblist.tsv"
ctr "$PYTHON" - "$ORDER_SEED" "$N_REPS" "$STEPS" "${DATA_FILES[@]}" > "$JOBLIST" <<'PY'
import random, sys
seed, reps, steps, files = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(), sys.argv[4:]
jobs = [(f, s, r) for f in files for s in steps for r in range(1, reps + 1)]
random.Random(seed).shuffle(jobs)
print("\n".join("\t".join(map(str, j)) for j in jobs))
PY

wait_for_idle() {
    [[ -n "$MAX_LOAD" ]] || return 0
    while awk -v max="$MAX_LOAD" '{ exit !($1 >= max) }' /proc/loadavg; do
        echo "    load $(cut -d' ' -f1 /proc/loadavg) >= $MAX_LOAD, waiting..."
        sleep 60
    done
}

TOTAL=$(wc -l < "$JOBLIST"); i=0
while IFS=$'\t' read -r data_file step rep; do
    i=$((i + 1))
    tag=$(basename "$data_file" .h5mu)
    name="${tag}__crx.${step}__rep${rep}"
    out="$RAW_DIR/$name.tsv"
    if [[ -s "$out" ]] && ! grep -q $'\terror\t' "$out"; then
        echo "[$i/$TOTAL] skip $name"; continue
    fi
    wait_for_idle
    echo "[$i/$TOTAL] $(date '+%F %T') $name"
    set +e
    timeout "$RUN_TIMEOUT" taskset -c "$CORES" "$RUNTIME" exec --cleanenv --bind "$BINDS" \
        --pwd "$PROJECT_ROOT" "$SIF" \
        "$PYTHON" -m supplementary.benchmark.benchmark_crx \
        --data_file "$data_file" --pseudotime_file "$PSEUDOTIME_DIR/pseudotime_${tag}.tsv" \
        --step "$step" --rep "$rep" --out_file "$out" \
        --dataset "$DATASET" --n_jobs "$N_JOBS" --cluster_key "$CLUSTER_KEY" \
        --n_states "$N_STATES" --n_components "$N_COMPONENTS" --use_petsc "$USE_PETSC" --allow_overlap "$ALLOW_OVERLAP" \
        > "$LOG_DIR/$name.log" 2>&1 < /dev/null   # stdin from /dev/null: do not eat the job list
    rc=$?
    set -e
    if (( rc != 0 )); then
        # 124 = timeout, 137 = killed (usually out of memory)
        printf "%s\t%s\t%s\t%s\n" "${data_file##*/}" "$step" "$rep" "$rc" >> "$FAILURES"
        echo "    failed with exit code $rc (see $LOG_DIR/$name.log)"
    fi
done < "$JOBLIST"

ctr "$PYTHON" - "$RAW_DIR" "$OUT_DIR/resources_crx.tsv" <<'PY'
import glob, sys
import pandas as pd
files = sorted(glob.glob(f"{sys.argv[1]}/*.tsv"))
df = pd.concat((pd.read_csv(f, sep="\t") for f in files), ignore_index=True)
df.sort_values(["function", "n_cells", "rep"]).to_csv(sys.argv[2], sep="\t", index=False)
ok = df[df.status == "ok"]
print(ok.groupby(["function", "n_cells"])[["wall_time_s", "peak_delta_gb"]].median().round(3).to_string())
print(f"\n{len(ok)}/{len(df)} runs ok -> {sys.argv[2]}")
if "n_macrostates" in ok:
    got = ok.dropna(subset=["n_macrostates"])
    print("\nstates found (macro / initial / terminal):")
    print(got.groupby("n_cells")[["n_macrostates", "n_initial_states", "n_terminal_states"]]
          .agg(lambda x: "/".join(map(str, sorted(set(x.astype(int)))))).to_string())
if (df.loadavg_1m_at_start > df.cpus_available).any():
    print("warning: some runs started while the server load exceeded the allotted cores")
PY
