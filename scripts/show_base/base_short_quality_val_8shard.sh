#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Direct child only of:
#   /tmp/globaldiff_guarded_runner.py --gpus 0,1,2,3,4,5,6,7 -- ...
#
# The preflight is produced by base_short_quality_val_adapter.py and contains
# exactly e1/e2/e4/e8.  Each candidate uses eight exact modulo shards.  The
# launcher publishes prediction, distribution, released2 screen, and raw
# replay artifacts; it never accepts a test path.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON PREFLIGHT PREFLIGHT_SHA RUN_ROOT SOURCE_COMMIT SOURCE_TREE TALKSHOW_ROOT FEATURE_EXTRACTOR SMPLX_ASSET REAL_CACHE REAL_CACHE_SHA REAL_CACHE_BYTES REAL_CACHE_PAYLOAD E1_GATE E1_GATE_SHA E2_GATE E2_GATE_SHA E4_GATE E4_GATE_SHA E8_GATE E8_GATE_SHA"
}

if [[ $# -ne 22 ]]; then
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
metric_root=$8
feature_extractor=$9
smplx_asset=${10}
real_cache=${11}
real_cache_sha=${12}
real_cache_bytes=${13}
real_cache_payload=${14}
gate1=${15}
gate1_sha=${16}
gate2=${17}
gate2_sha=${18}
gate4=${19}
gate4_sha=${20}
gate8=${21}
gate8_sha=${22}

for path in "$repo_root" "$python_bin" "$preflight" "$run_root" \
    "$metric_root" "$feature_extractor" "$smplx_asset" "$real_cache" \
    "$gate1" "$gate2" "$gate4" "$gate8"; do
    if [[ "$path" != /* ]]; then
        printf 'all paths must be absolute: %s\n' "$path" >&2
        exit 2
    fi
done
for digest in "$preflight_sha" "$real_cache_sha" "$real_cache_payload" \
    "$gate1_sha" "$gate2_sha" "$gate4_sha" "$gate8_sha"; do
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
if [[ ! "$real_cache_bytes" =~ ^[1-9][0-9]*$ ]]; then
    printf 'real-cache byte count must be positive\n' >&2
    exit 2
fi
if [[ ! -x "$python_bin" || ! -f "$preflight" || -L "$preflight" || \
      -e "$run_root" || -L "$run_root" || ! -d "$(dirname "$run_root")" ]]; then
    printf 'unsafe Python, preflight, or run root\n' >&2
    exit 1
fi

adapter="$repo_root/scripts/show_base/base_short_quality_val_adapter.py"
replay="$repo_root/scripts/show_base/replay_released2_primary.py"
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
for required in "$adapter" "$replay" \
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
    scripts/show_base/replay_released2_primary.py \
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

# The validated preflight is safe to project.  NUL delimiters preserve exact
# canonical paths and avoid shell word splitting.
mapfile -d '' -t common < <(
    "$python_bin" - "$preflight" "$preflight_sha" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = path.read_bytes()
if hashlib.sha256(payload).hexdigest() != sys.argv[2]:
    raise SystemExit("preflight changed after validation")
value = json.loads(payload)
canonical = json.loads(Path(value["val_inputs_receipt"]["path"]).read_bytes())["canonical_manifest"]
fields = [canonical["path"], canonical["sha256"]]
canonical_payload = Path(canonical["path"]).read_bytes()
fields.append(str(len(canonical_payload)))
for field in fields:
    sys.stdout.write(str(field))
    sys.stdout.write("\0")
PY
)
if [[ ${#common[@]} -ne 3 || ! ${common[1]} =~ ^[0-9a-f]{64}$ || \
      ! ${common[2]} =~ ^[1-9][0-9]*$ ]]; then
    printf 'canonical val projection failed\n' >&2
    exit 1
fi
canonical_manifest=${common[0]}
canonical_manifest_sha=${common[1]}
canonical_manifest_bytes=${common[2]}

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

declare -A gate_path gate_sha
gate_path[1]=$gate1; gate_sha[1]=$gate1_sha
gate_path[2]=$gate2; gate_sha[2]=$gate2_sha
gate_path[4]=$gate4; gate_sha[4]=$gate4_sha
gate_path[8]=$gate8; gate_sha[8]=$gate8_sha

for epoch in 1 2 4 8; do
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
    manifest="$candidate_root/final/final_manifest.jsonl"
    artifact_fields "$lineage" true
    lineage_sha=${artifact_result[0]}
    artifact_fields "$manifest"
    manifest_sha=${artifact_result[0]}
    manifest_bytes=${artifact_result[1]}

    distribution="$candidate_root/distribution.json"
    "$python_bin" "$adapter" distribution --split val \
        --preflight "$preflight" --expected-preflight-sha256 "$preflight_sha" \
        --epoch "$epoch" --lineage "$lineage" \
        --expected-lineage-sha256 "$lineage_sha" \
        --validation-gate "${gate_path[$epoch]}" \
        --expected-validation-gate-sha256 "${gate_sha[$epoch]}" \
        --output "$distribution" \
        >"$run_root/logs/e${epoch}-distribution.log" 2>&1
    artifact_fields "$distribution" true
    distribution_sha=${artifact_result[0]}
    distribution_bytes=${artifact_result[1]}
    distribution_payload=${artifact_result[2]}

    screen="$candidate_root/released2-primary-screen.json"
    command=(
        "$python_bin" "$replay" screen
        --talkshow-metric-root "$metric_root"
        --feature-extractor "$feature_extractor" --smplx-asset "$smplx_asset"
        --device cuda:0 --split val --expected-clip-count 1715
        --canonical-manifest "$canonical_manifest"
        --expected-canonical-manifest-sha256 "$canonical_manifest_sha"
        --expected-canonical-manifest-bytes "$canonical_manifest_bytes"
        --prediction-manifest "$manifest"
        --expected-prediction-manifest-sha256 "$manifest_sha"
        --expected-prediction-manifest-bytes "$manifest_bytes"
        --distribution-json "$distribution"
        --expected-distribution-sha256 "$distribution_sha"
        --expected-distribution-bytes "$distribution_bytes"
        --expected-distribution-payload-sha256 "$distribution_payload"
        --real-feature-cache-json "$real_cache"
        --expected-real-feature-cache-sha256 "$real_cache_sha"
        --expected-real-feature-cache-bytes "$real_cache_bytes"
        --expected-real-feature-cache-payload-sha256 "$real_cache_payload"
        --output-json "$screen"
    )
    launch_registered "$run_root/logs/e${epoch}-screen.log" "${command[@]}"
    wait "$LAST_CHILD_PID"
    artifact_fields "$screen" true
    screen_sha=${artifact_result[0]}
    screen_bytes=${artifact_result[1]}

    raw_replay="$candidate_root/released2-primary-raw-replay.json"
    command=(
        "$python_bin" "$replay" replay
        --talkshow-metric-root "$metric_root"
        --feature-extractor "$feature_extractor" --smplx-asset "$smplx_asset"
        --device cuda:0 --split val --expected-clip-count 1715
        --report-json "$screen" --expected-report-sha256 "$screen_sha"
        --expected-report-bytes "$screen_bytes"
        --prediction-manifest "$manifest"
        --expected-prediction-manifest-sha256 "$manifest_sha"
        --expected-prediction-manifest-bytes "$manifest_bytes"
        --real-feature-cache-json "$real_cache"
        --expected-real-feature-cache-sha256 "$real_cache_sha"
        --expected-real-feature-cache-bytes "$real_cache_bytes"
        --expected-real-feature-cache-payload-sha256 "$real_cache_payload"
        --output-json "$raw_replay"
    )
    launch_registered "$run_root/logs/e${epoch}-raw-replay.log" "${command[@]}"
    wait "$LAST_CHILD_PID"
    artifact_fields "$raw_replay" true
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
for epoch in (1, 2, 4, 8):
    candidate = root / "candidates" / f"e{epoch}"
    artifacts = {}
    for name, relative in (
        ("prediction_manifest", "final/final_manifest.jsonl"),
        ("distribution_receipt", "distribution.json"),
        ("primary_screen_receipt", "released2-primary-screen.json"),
        ("primary_replay_receipt", "released2-primary-raw-replay.json"),
    ):
        path = candidate / relative
        data = path.read_bytes()
        artifact = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if path.suffix == ".json":
            artifact["receipt_payload_sha256"] = json.loads(data)["receipt_payload_sha256"]
        artifacts[name] = artifact
    rows.append({"epoch": epoch, **artifacts})
body = {
    "format": "semtalk_show_base_short_quality_val_completion_v1",
    "status": "complete",
    "split": "val",
    "test_visible": False,
    "candidate_epochs": [1, 2, 4, 8],
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
