#!/usr/bin/env bash
# Full repository benchmark. Run from the cluster login node: bash benchmark.sh
# guidlines.html is truncated; defaults below come from the repository and the
# supplied A100 srun command, not an inferred assignment grading configuration.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
runs=3
warmups=1
gpu=a100-80
cpus=1
memory=20G
limit=1800
local_run=0
build=1
dry_run=0
out=""
programs=()
scenes=()

usage() {
    cat <<'HELP'
Usage: bash benchmark.sh [options]
  --runs N          Measured runs per program/scene (default: 3)
  --warmups N       Untimed warmup runs per pair (default: 1)
  --gpu TYPE        Slurm GPU type (default: a100-80)
  --cpus N          Allocated CPUs and OMP_NUM_THREADS (default: 1)
  --mem SIZE        Slurm memory (default: 20G)
  --timeout N       Seconds per invocation; 0 disables (default: 1800)
  --out DIR         New results directory (must not already exist)
  --program NAME    Select program; repeatable. Default: path_tracing,
                    cpu_path_tracing, bench_0, ..., bench_4
  --scene NAME      Select scene basename; repeatable. Default: all 8 scenes
  --local           Run on current node (use inside an existing allocation)
  --no-build        Use existing binaries instead of force-rebuilding sources
  --dry-run         Print build/run commands without executing or writing files
  --help            Show help

Default: one srun allocation, sequential execution, unchanged scene settings,
correctness checks on every invocation. Requires bash, python3, make/nvcc and
GNU timeout (unless --timeout 0). Results: runs.tsv, summary.tsv, logs/, images/.
Nonzero exit if any invocation fails, times out, or lacks a correctness PASS.
HELP
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --local) local_run=1; shift ;;
        --no-build) build=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        --runs|--warmups|--gpu|--cpus|--mem|--timeout|--out|--program|--scene)
            (($# >= 2)) || fail "Missing value for $1"
            case "$1" in
                --runs) runs=$2 ;; --warmups) warmups=$2 ;;
                --gpu) gpu=$2 ;; --cpus) cpus=$2 ;; --mem) memory=$2 ;;
                --timeout) limit=$2 ;; --out) out=$2 ;;
                --program) programs+=("$2") ;; --scene) scenes+=("$2") ;;
            esac
            shift 2 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
for value in "$runs" "$warmups" "$cpus" "$limit"; do
    [[ $value =~ ^(0|[1-9][0-9]*)$ ]] || fail "Counts must be nonnegative decimal integers"
done
((runs > 0 && cpus > 0)) || fail 'Runs and CPUs must be positive'
((${#programs[@]})) || programs=(path_tracing cpu_path_tracing bench_0 bench_1 bench_2 bench_3 bench_4)
((${#scenes[@]})) || scenes=(debug_0 debug_1 debug_2 cornell_box cornell_box_2 chess dragon ufo)
[[ -n $out ]] || out="benchmark-results/$(date +%Y%m%d-%H%M%S)-$$"
[[ $out = /* ]] || out="$ROOT/$out"
[[ ! -e $out ]] || fail "Results directory already exists: $out"

binary_for() {
    case "$1" in
        path_tracing|cpu_path_tracing) printf './%s' "$1" ;;
        bench_[0-4]) printf './benchmarks/%s' "$1" ;;
        *) fail "Unknown program: $1" ;;
    esac
}
scene_for() {
    case "$1" in
        debug_[0-2]) printf 'scenes/debug/%s.rtscene' "$1" ;;
        cornell_box|cornell_box_2|chess|dragon|ufo) printf 'scenes/%s.rtscene' "$1" ;;
        *) fail "Unknown scene: $1" ;;
    esac
}
for program in "${programs[@]}"; do binary_for "$program" >/dev/null; done
for scene in "${scenes[@]}"; do
    scene_path=$(scene_for "$scene")
    [[ -f $scene_path && -f images/$scene.ppm ]] || fail "Missing scene/reference for $scene"
done
print_command() { printf '%q ' "$@"; printf '\n'; }

# Relaunch the entire suite once, so runs share the same node/GPU allocation.
if (( ! local_run )); then
    worker=(bash "$ROOT/benchmark.sh" --local --runs "$runs" --warmups "$warmups"
            --cpus "$cpus" --timeout "$limit" --out "$out")
    ((build)) || worker+=(--no-build)
    (( ! dry_run )) || worker+=(--dry-run)
    for program in "${programs[@]}"; do worker+=(--program "$program"); done
    for scene in "${scenes[@]}"; do worker+=(--scene "$scene"); done
    launch=(srun --ntasks=1 --cpus-per-task="$cpus" --cpu-bind=core
            --mem="$memory" --gpus="$gpu")
    if ((dry_run)); then
        print_command "${launch[@]}" "${worker[@]}"
        exec "${worker[@]}"
    fi
    command -v srun >/dev/null || fail 'srun not found; use --local on a GPU machine'
    exec "${launch[@]}" "${worker[@]}"
fi

export OMP_NUM_THREADS="$cpus"
export LC_ALL=C
build_targets=()
for program in "${programs[@]}"; do
    case "$program" in path_tracing|cpu_path_tracing) build_targets+=("$program");; esac
done
if ((dry_run)); then
    if ((build && ${#build_targets[@]})); then print_command make -B "${build_targets[@]}"; fi
else
    command -v python3 >/dev/null || fail 'python3 is required'
    if ((limit > 0)); then command -v timeout >/dev/null || fail 'GNU timeout is required, or use --timeout 0'; fi
    mkdir -p "$out/logs" "$out/images"
    {
        date -u
        uname -a
        printf 'OMP_NUM_THREADS=%s\nSLURM_JOB_ID=%s\n' "$cpus" "${SLURM_JOB_ID:-none}"
        printf 'runs=%s warmups=%s timeout=%s\n' "$runs" "$warmups" "$limit"
        git rev-parse HEAD 2>/dev/null || true
        git status --short 2>/dev/null || true
        nvidia-smi 2>&1 || true
        nvcc --version 2>&1 || true
    } >"$out/environment.txt"
    if ((build && ${#build_targets[@]})); then
        make -B "${build_targets[@]}" 2>&1 | tee "$out/build.log"
    fi
    for program in "${programs[@]}"; do
        binary=$(binary_for "$program")
        [[ -x $binary ]] || fail "Missing executable: $binary"
        file "$binary" >>"$out/environment.txt" 2>/dev/null || true
    done
    printf 'program\tscene\tphase\trun\texit_code\tseconds\terror\tmax_error\tstatus\n' >"$out/runs.tsv"
fi

failed=0
for scene in "${scenes[@]}"; do
    scene_path=$(scene_for "$scene")
    for program in "${programs[@]}"; do
        binary=$(binary_for "$program")
        for ((iteration=1; iteration<=warmups+runs; ++iteration)); do
            phase=measured
            index=$((iteration - warmups))
            if ((iteration <= warmups)); then phase=warmup; index=$iteration; fi
            tag="$scene-$program-$phase-$index"
            log="$out/logs/$tag.log"
            # Overwrite only this pair's image to avoid storing every repeat.
            command=("$binary" "$scene_path" --correctness-check "images/$scene.ppm"
                     --output "$out/images/$scene-$program.ppm")
            if ((limit > 0)); then command=(timeout --kill-after=30s "${limit}s" "${command[@]}"); fi
            if ((dry_run)); then print_command "${command[@]}"; continue; fi
            printf '[%s / %s] %s %s\n' "$scene" "$program" "$phase" "$index"
            code=0
            "${command[@]}" >"$log" 2>&1 || code=$?
            # Parse the renderer's own timed region; exclude Slurm queue time,
            # scene loading, image writing, and correctness comparison.
            row=$(python3 - "$log" "$code" <<'PY'
import math, re, sys
text = open(sys.argv[1], errors="replace").read()
code = int(sys.argv[2])
def field(pattern):
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else ""
seconds = field(r"^Time taken:\s*(\S+) seconds\s*$")
error = field(r"^Error:\s*(\S+)\s*$")
maximum = field(r"^Max error:\s*(\S+)\s*$")
try:
    valid = math.isfinite(float(seconds)) and float(seconds) > 0
except ValueError:
    valid = False
passed = bool(re.search(r"^Correctness check: PASS\s*$", text, re.M))
status = "PASS" if code == 0 and valid and passed else "FAIL"
if code in (124, 137):
    status = "TIMEOUT"
print("\t".join((str(code), seconds, error, maximum, status)))
PY
)
            printf '%s\t%s\t%s\t%s\t%s\n' "$program" "$scene" "$phase" "$index" "$row" >>"$out/runs.tsv"
            printf '  %s\n' "$row"
            if [[ $row != *$'\tPASS' ]]; then
                failed=1
                printf '  See %s\n' "$log" >&2
            fi
        done
    done
done
(( ! dry_run )) || exit 0

python3 - "$out" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
out = Path(sys.argv[1])
groups = defaultdict(list)
with (out / "runs.tsv").open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["phase"] == "measured":
            groups[row["scene"], row["program"]].append(row)
with (out / "summary.tsv").open("w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(("scene", "program", "passed", "total", "mean_seconds",
                     "median_seconds", "min_seconds", "max_seconds"))
    for (scene, program), rows in groups.items():
        times = [float(r["seconds"]) for r in rows if r["status"] == "PASS"]
        # Never present partial/failed configurations as successful timings.
        stats = (statistics.mean(times), statistics.median(times), min(times), max(times)) if len(times) == len(rows) else ()
        values = [f"{v:.9g}" for v in stats] if stats else [""] * 4
        writer.writerow((scene, program, len(times), len(rows), *values))
print((out / "summary.tsv").read_text())
PY
printf 'Results saved to %s\n' "$out"
exit "$failed"
