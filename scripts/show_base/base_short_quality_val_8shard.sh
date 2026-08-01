#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Direct child only of:
#   /tmp/globaldiff_guarded_runner.py --gpus 0,1,2,3,4,5,6,7 -- ...
#
# The preflight is produced by base_short_quality_val_adapter.py and contains
# exactly e1/e2/e4/e8/e16/e32.  Each candidate uses eight exact modulo shards.
# launcher publishes the formal inference lineage and pinned DiffSHEG SHOW
# validation FGD report; it never accepts a test path.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON PREFLIGHT PREFLIGHT_SHA RUN_ROOT SOURCE_COMMIT SOURCE_TREE PASPA_ROOT DIFFSHEG_ROOT DIFFSHEG_BATCH_SIZE"
}

if [[ $# -ne 10 ]]; then
    usage >&2
    exit 2
fi

repo_root=$1
python_bin=$2
preflight=$3
preflight_sha=$4
run_root=$5
source_commit=$6
source_tree=$7
paspa_root=$8
diffsheg_root=$9
diffsheg_batch_size=${10}

for path in "$repo_root" "$python_bin" "$preflight" "$run_root" \
    "$paspa_root" "$diffsheg_root"; do
    if [[ "$path" != /* ]]; then
        printf 'all paths must be absolute: %s\n' "$path" >&2
        exit 2
    fi
done
for digest in "$preflight_sha"; do
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'invalid SHA-256 argument\n' >&2
        exit 2
    fi
done
for oid in "$source_commit" "$source_tree"; do
    if [[ ! "$oid" =~ ^[0-9a-f]{40}$ ]]; then
        printf 'invalid Git object ID\n' >&2
        exit 2
    fi
done
if [[ ! "$diffsheg_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    printf 'DiffSHEG batch size must be positive\n' >&2
    exit 2
fi
if [[ ! -x "$python_bin" || ! -f "$preflight" || -L "$preflight" || \
      -e "$run_root" || -L "$run_root" || ! -d "$(dirname "$run_root")" ]]; then
    printf 'unsafe Python, preflight, or run root\n' >&2
    exit 1
fi

adapter="$repo_root/scripts/show_base/base_short_quality_val_adapter.py"
evaluator="$repo_root/scripts/show_base/evaluate_diffsheg_val_fgd.py"
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
for required in "$adapter" "$evaluator" \
    "$launcher_dir/guarded_runner_contract.sh"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        printf 'required source is missing or symlinked: %s\n' "$required" >&2
        exit 1
    fi
done
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

if [[ "$(git -C "$repo_root" remote get-url origin)" != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      "$(git -C "$repo_root" remote get-url --push origin)" != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      "$(git -C "$repo_root" rev-parse HEAD)" != "$source_commit" || \
      "$(git -C "$repo_root" rev-parse 'HEAD^{tree}')" != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref \
          --format='%(objectname)' refs/heads | awk -v head="$source_commit" '$1 == head')" ]]; then
    printf 'source must be exact clean detached SemTalk with no local head at commit\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/base_short_quality_val_adapter.py \
    scripts/show_base/base_short_quality_val_8shard.sh \
    scripts/show_base/run_base_val_inference.py \
    scripts/show_base/semtalk_base_inference_core.py \
    scripts/show_base/base_long_val_contract.py \
    scripts/show_base/select_base_official_adapt.py \
    scripts/show_base/evaluate_diffsheg_val_fgd.py \
    scripts/show_base/guarded_runner_contract.sh; do
    if [[ "$(git -C "$repo_root" ls-files --error-unmatch "$tracked")" != \
          "$tracked" ]]; then
        printf 'formal source is not tracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

mkdir "$run_root"
mkdir "$run_root/logs" "$run_root/candidates"
"$python_bin" "$adapter" validate --split val \
    --preflight "$preflight" --expected-preflight-sha256 "$preflight_sha" \
    >"$run_root/logs/preflight-replay.log" 2>&1

artifact_fields() {
    local path=$1
    local payload=${2:-false}
    mapfile -d '' -t artifact_result < <(
        "$python_bin" - "$path" "$payload" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
    raise SystemExit("artifact path is unsafe")
data = path.read_bytes()
fields = [hashlib.sha256(data).hexdigest(), str(len(data))]
if sys.argv[2] == "true":
    fields.append(json.loads(data)["receipt_payload_sha256"])
for field in fields:
    sys.stdout.write(str(field))
    sys.stdout.write("\0")
PY
    )
    if [[ ${#artifact_result[@]} -lt 2 || \
          ! ${artifact_result[0]} =~ ^[0-9a-f]{64}$ || \
          ! ${artifact_result[1]} =~ ^[1-9][0-9]*$ ]]; then
        printf 'cannot pin artifact: %s\n' "$path" >&2
        exit 1
    fi
}

registered_pids=()
registered_starts=()
registered_argv_sha=()

proc_start() {
    local line rest state
    local -a fields
    line=$(<"/proc/$1/stat")
    rest=${line##*) }
    read -r -a fields <<<"$rest"
    state=${fields[0]}
    [[ "$state" != Z && ${fields[19]} =~ ^[0-9]+$ ]] || return 1
    printf '%s' "${fields[19]}"
}

capture_fork_start() {
    local pid=$1 observed_parent observed_start
    for _attempt in {1..400}; do
        if [[ -r "/proc/$pid/status" && -r "/proc/$pid/stat" ]]; then
            observed_parent=$(awk '/^PPid:/{print $2}' "/proc/$pid/status")
            observed_start=$(proc_start "$pid" || true)
            if [[ "$observed_parent" == "$$" && "$observed_start" =~ ^[0-9]+$ ]]; then
                printf '%s' "$observed_start"
                return 0
            fi
        else
            return 1
        fi
        sleep 0.01
    done
    return 1
}

register_pid() {
    local pid=$1
    shift
    local -a identity
    mapfile -t identity < <(
        "$python_bin" - "$pid" "$$" "$@" <<'PY'
import hashlib
import os
from pathlib import Path
import sys
import time

pid = int(sys.argv[1])
expected_ppid = int(sys.argv[2])
expected_argv = [os.fsencode(value) for value in sys.argv[3:]]
proc = Path("/proc") / str(pid)
for _ in range(400):
    try:
        stat_line = (proc / "stat").read_text()
        cmdline = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        raise SystemExit(f"child {pid} exited before exact argv capture")
    fields = stat_line.rsplit(") ", 1)[1].split()
    argv = [token for token in cmdline.split(b"\0") if token]
    if fields[0] != "Z" and int(fields[1]) == expected_ppid and argv == expected_argv:
        print(fields[19])
        print(hashlib.sha256(cmdline).hexdigest())
        raise SystemExit(0)
    time.sleep(0.01)
raise SystemExit(f"child {pid} never reached exact registered argv")
PY
    )
    if [[ ${#identity[@]} -ne 2 || ! ${identity[0]} =~ ^[0-9]+$ || \
          ! ${identity[1]} =~ ^[0-9a-f]{64}$ ]]; then
        printf 'child identity/parent/full argv mismatch: %s\n' "$pid" >&2
        return 1
    fi
    registered_pids+=("$pid")
    registered_starts+=("${identity[0]}")
    registered_argv_sha+=("${identity[1]}")
}

cleanup_registered() {
    local index pid observed_start observed_sha observed_parent
    trap - EXIT INT TERM
    for ((index = 0; index < ${#registered_pids[@]}; index++)); do
        pid=${registered_pids[$index]}
        [[ -r "/proc/$pid/status" && -r "/proc/$pid/cmdline" ]] || continue
        observed_parent=$(awk '/^PPid:/{print $2}' "/proc/$pid/status")
        observed_start=$(proc_start "$pid" || true)
        observed_sha=$(sha256sum "/proc/$pid/cmdline" | awk '{print $1}')
        if [[ "$observed_parent" == "$$" && \
              "$observed_start" == "${registered_starts[$index]}" && \
              "$observed_sha" == "${registered_argv_sha[$index]}" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${registered_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup_registered EXIT INT TERM

launch_registered() {
    local log_path=$1
    shift
    local -a command=("$@")
    "${command[@]}" >"$log_path" 2>&1 &
    local pid=$! start
    start=$(capture_fork_start "$pid" || true)
    if [[ ! "$start" =~ ^[0-9]+$ ]] || ! register_pid "$pid" "${command[@]}"; then
        if [[ "$start" =~ ^[0-9]+$ && -r "/proc/$pid/status" && \
              "$(awk '/^PPid:/{print $2}' "/proc/$pid/status")" == "$$" && \
              "$(proc_start "$pid" || true)" == "$start" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
        return 1
    fi
    LAST_CHILD_PID=$pid
}

for epoch in 1 2 4 8 16 32; do
    candidate_root="$run_root/candidates/e$epoch"
    mkdir "$candidate_root"
    shard_pids=()
    for shard_id in 0 1 2 3 4 5 6 7; do
        command=(
            "$python_bin" "$adapter" shard --split val
            --preflight "$preflight" --expected-preflight-sha256 "$preflight_sha"
            --epoch "$epoch" --output-root "$candidate_root"
            --num-shards 8 --shard-id "$shard_id"
            --device "cuda:$shard_id"
        )
        launch_registered "$run_root/logs/e${epoch}-shard${shard_id}.log" "${command[@]}"
        shard_pids+=("$LAST_CHILD_PID")
    done
    failed=0
    for pid in "${shard_pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [[ $failed -ne 0 ]]; then
        printf 'short-quality shard failure at e%s\n' "$epoch" >&2
        exit 1
    fi

    "$python_bin" "$adapter" finalize --split val \
        --preflight "$preflight" --expected-preflight-sha256 "$preflight_sha" \
        --epoch "$epoch" --output-root "$candidate_root" --num-shards 8 \
        >"$run_root/logs/e${epoch}-finalize.log" 2>&1
    lineage="$candidate_root/final/val-inference-lineage.json"
    clip_manifest="$candidate_root/final/diffsheg_eval_clip_ids.txt"
    artifact_fields "$lineage" true
    artifact_fields "$clip_manifest"
    clip_manifest_sha=${artifact_result[0]}

    report="$candidate_root/diffsheg-val-fgd.json"
    command=(
        "$python_bin" "$evaluator"
        --pred-dir "$candidate_root/final/predictions/val"
        --gt-dir "$candidate_root/final/ground-truth/val"
        --clip-manifest "$clip_manifest"
        --clip-manifest-sha256 "$clip_manifest_sha"
        --paspa-root "$paspa_root" --diffsheg-root "$diffsheg_root"
        --device cuda:0 --batch-size "$diffsheg_batch_size"
        --output "$report"
    )
    launch_registered "$run_root/logs/e${epoch}-diffsheg-fgd.log" "${command[@]}"
    if ! wait "$LAST_CHILD_PID"; then
        printf 'DiffSHEG validation FGD failed for e%s\n' "$epoch" >&2
        exit 1
    fi
    artifact_fields "$report"
done

"$python_bin" - "$run_root" "$preflight" "$preflight_sha" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
preflight = Path(sys.argv[2])
rows = []
for epoch in (1, 2, 4, 8, 16, 32):
    candidate = root / "candidates" / f"e{epoch}"
    artifacts = {}
    for name, relative in (
        ("inference_lineage", "final/val-inference-lineage.json"),
        ("diffsheg_report", "diffsheg-val-fgd.json"),
    ):
        path = candidate / relative
        data = path.read_bytes()
        artifact = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if name == "inference_lineage":
            artifact["receipt_payload_sha256"] = json.loads(data)["receipt_payload_sha256"]
        artifacts[name] = artifact
    rows.append({"epoch": epoch, **artifacts})
body = {
    "format": "semtalk_show_base_short_quality_val_completion_v2",
    "status": "complete",
    "split": "val",
    "test_visible": False,
    "candidate_epochs": [1, 2, 4, 8, 16, 32],
    "preflight": {"path": str(preflight), "sha256": sys.argv[3]},
    "candidates": rows,
}
raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
body["receipt_payload_sha256"] = hashlib.sha256(raw).hexdigest()
payload = (json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
output = root / "completion.json"
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(fd, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY

trap - EXIT INT TERM
exit 0
