#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Direct child only of:
#   /tmp/globaldiff_guarded_runner.py --gpus 0,1,2,3,4,5,6,7 -- ...
#
# A worker owns one static candidate-index partition.  With two workers use
# PARTITION_ID/PARTITION_COUNT 0/2 and 1/2: each worker evaluates eleven
# candidates; every candidate still uses all eight physical GPUs, one exact
# modulo shard per GPU.  The CPU-only union publisher rejects overlap,
# omission, path reuse, source/runtime/input disagreement, or any test input.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON RUN_SPEC RUN_SPEC_SHA256 RUN_SPEC_PAYLOAD_SHA256 PARTITION_ID PARTITION_COUNT RUN_ROOT SOURCE_COMMIT SOURCE_TREE"
}

if [[ $# -ne 10 ]]; then
    usage >&2
    exit 2
fi

repo_root=$1
python_bin=$2
run_spec=$3
run_spec_sha256=$4
run_spec_payload_sha256=$5
partition_id=$6
partition_count=$7
run_root=$8
source_commit=$9
source_tree=${10}

if [[ ! "$repo_root" = /* || ! "$python_bin" = /* || \
      ! "$run_spec" = /* || ! "$run_root" = /* ]]; then
    printf 'all paths must be absolute\n' >&2
    exit 2
fi
if [[ "$partition_count" != 2 || \
      ( "$partition_id" != 0 && "$partition_id" != 1 ) ]]; then
    printf 'formal run requires exact static partition 0/2 or 1/2\n' >&2
    exit 2
fi
formal_host=$(hostname)
case "$partition_id" in
    0)
        expected_formal_host=
        expected_formal_host+=iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0
        ;;
    1)
        expected_formal_host=
        expected_formal_host+=iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0
        ;;
esac
for digest in "$run_spec_sha256" "$run_spec_payload_sha256"; do
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
if [[ ! -x "$python_bin" || ! -f "$run_spec" || -L "$run_spec" ]]; then
    printf 'Python or run specification is unsafe\n' >&2
    exit 1
fi
if [[ -e "$run_root" || -L "$run_root" || \
      ! -d "$(dirname "$run_root")" ]]; then
    printf 'formal run root must be new below an existing parent\n' >&2
    exit 1
fi

orchestrator="$repo_root/scripts/show_base/base_fresh_val_orchestrator.py"
inference="$repo_root/scripts/show_base/run_base_val_inference.py"
metrics="$repo_root/scripts/show_base/evaluate_talkshow_show_metrics.py"
replay="$repo_root/scripts/show_base/replay_released2_primary.py"
selector="$repo_root/scripts/show_base/select_published_base_winner.py"
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
for required in "$orchestrator" "$inference" "$metrics" "$replay" \
    "$selector"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        printf 'required source is missing or symlinked: %s\n' "$required" >&2
        exit 1
    fi
done

# The shared contract derives the exact direct parent from Bash's read-only
# PPID and fail-closes on Python argv[1], a unique runner-side `--` delimiter,
# and one exact all-GPU reservation before that delimiter.  Workload-side
# runner/GPU-looking tokens are inert and cannot grant authority.
if [[ ! -f "$launcher_dir/guarded_runner_contract.sh" || \
      -L "$launcher_dir/guarded_runner_contract.sh" ]]; then
    printf 'guarded runner contract is unavailable\n' >&2
    exit 1
fi
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus
if [[ "$formal_host" != "$expected_formal_host" ]]; then
    printf 'partition %s requires exact formal host %s, observed %s\n' \
        "$partition_id" "$expected_formal_host" "$formal_host" >&2
    exit 1
fi

if [[ "$(git -C "$repo_root" remote get-url origin)" != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      "$(git -C "$repo_root" rev-parse HEAD)" != "$source_commit" || \
      "$(git -C "$repo_root" rev-parse 'HEAD^{tree}')" != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'formal source must be exact clean detached SemTalk with zero local branches\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/guarded_runner_contract.sh \
    scripts/show_base/base_fresh_val_orchestrator.py \
    scripts/show_base/run_base_val_inference.py \
    scripts/show_base/evaluate_talkshow_show_metrics.py \
    scripts/show_base/replay_released2_primary.py \
    scripts/show_base/select_published_base_winner.py \
    scripts/show_base/published_test_winner_claim.py; do
    if [[ "$(git -C "$repo_root" ls-files --error-unmatch "$tracked")" != \
          "$tracked" ]]; then
        printf 'formal source is not tracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

mapfile -d '' -t run_root_identity < <(
    "$python_bin" "$orchestrator" create-run-root --path "$run_root" \
        --subdirectory logs --subdirectory candidates --subdirectory seals
)
if [[ ${#run_root_identity[@]} -ne 3 || \
      "${run_root_identity[0]}" != "$run_root" || \
      ! ${run_root_identity[1]} =~ ^[0-9]+$ || \
      ! ${run_root_identity[2]} =~ ^[0-9]+$ ]]; then
    printf 'formal run root could not be created safely\n' >&2
    exit 1
fi
spec_audit="$run_root/run-spec-audit.json"
"$python_bin" "$orchestrator" validate-run-spec \
    --run-spec-path "$run_spec" \
    --run-spec-sha256 "$run_spec_sha256" \
    --run-spec-payload-sha256 "$run_spec_payload_sha256" \
    --source-commit "$source_commit" --source-tree "$source_tree" \
    --output-json "$spec_audit"

# NUL-delimited extraction keeps every frozen path exact.  validate-run-spec
# already verified all bytes, hashes, source, split and 22 per-candidate gates.
mapfile -d '' -t common < <(
    "$python_bin" "$orchestrator" extract-run-spec \
        --run-spec-path "$run_spec" \
        --run-spec-sha256 "$run_spec_sha256" \
        --run-spec-payload-sha256 "$run_spec_payload_sha256" \
        --source-commit "$source_commit" --source-tree "$source_tree" \
        --profile launcher
)
if [[ ${#common[@]} -ne 30 ]]; then
    printf 'validated run-spec extraction failed\n' >&2
    exit 1
fi

candidate_manifest=${common[0]}
candidate_manifest_sha=${common[1]}
candidate_status=${common[2]}
candidate_status_sha=${common[3]}
frozen_inputs=${common[4]}
frozen_inputs_sha=${common[5]}
val_inputs=${common[6]}
val_inputs_sha=${common[7]}
pipeline=${common[8]}
pipeline_sha=${common[9]}
prerequisite=${common[10]}
prerequisite_sha=${common[11]}
prerequisite_bytes=${common[12]}
prerequisite_payload=${common[13]}
continuation=${common[14]}
continuation_sha=${common[15]}
continuation_bytes=${common[16]}
continuation_payload=${common[17]}
canonical_manifest=${common[18]}
canonical_manifest_sha=${common[19]}
canonical_manifest_bytes=${common[20]}
metric_root=${common[21]}
feature_extractor=${common[22]}
smplx_asset=${common[23]}
real_cache=${common[24]}
real_cache_sha=${common[25]}
real_cache_bytes=${common[26]}
real_cache_payload=${common[27]}
seed=${common[28]}
candidates_per_wave=${common[29]}
if [[ "$candidates_per_wave" != 1 && "$candidates_per_wave" != 2 && \
      "$candidates_per_wave" != 4 ]]; then
    printf 'validated candidates-per-wave is not 1, 2, or 4\n' >&2
    exit 1
fi

artifact_fields() {
    local path=$1
    local payload_flag=${2:-false}
    local -a fields_command=(
        "$python_bin" "$orchestrator" artifact-fields
        --artifact-path "$path"
    )
    if [[ "$payload_flag" == true ]]; then
        fields_command+=(--payload-receipt)
    fi
    mapfile -d '' -t artifact_result < <(
        "${fields_command[@]}"
    )
    if [[ ${#artifact_result[@]} -ne 3 || \
          ! ${artifact_result[0]} =~ ^[0-9a-f]{64}$ ]]; then
        printf 'cannot pin output artifact: %s\n' "$path" >&2
        exit 1
    fi
}

preflight="$run_root/preflight.json"
"$python_bin" "$inference" prepare --split val \
    --candidate-manifest "$candidate_manifest" \
    --expected-candidate-manifest-sha256 "$candidate_manifest_sha" \
    --candidate-status "$candidate_status" \
    --expected-candidate-status-sha256 "$candidate_status_sha" \
    --frozen-inputs "$frozen_inputs" \
    --expected-frozen-inputs-sha256 "$frozen_inputs_sha" \
    --val-inputs "$val_inputs" \
    --expected-val-inputs-sha256 "$val_inputs_sha" \
    --pipeline "$pipeline" \
    --expected-pipeline-sha256 "$pipeline_sha" \
    --output "$preflight" >"$run_root/logs/preflight.log" 2>&1
artifact_fields "$preflight" true
preflight_sha=${artifact_result[0]}
preflight_bytes=${artifact_result[1]}
preflight_payload=${artifact_result[2]}

plan="$run_root/plan.json"
"$python_bin" "$orchestrator" plan \
    --preflight-path "$preflight" --preflight-sha256 "$preflight_sha" \
    --preflight-payload-sha256 "$preflight_payload" \
    --prerequisite-path "$prerequisite" \
    --prerequisite-sha256 "$prerequisite_sha" \
    --prerequisite-payload-sha256 "$prerequisite_payload" \
    --continuation-path "$continuation" \
    --continuation-sha256 "$continuation_sha" \
    --continuation-payload-sha256 "$continuation_payload" \
    --output-json "$plan" >"$run_root/logs/plan.log" 2>&1

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
    local pid=$1
    local observed_parent observed_start
    for _attempt in {1..400}; do
        if [[ -r "/proc/$pid/status" && -r "/proc/$pid/stat" ]]; then
            observed_parent=$(awk '/^PPid:/{print $2}' "/proc/$pid/status")
            observed_start=$(proc_start "$pid" || true)
            if [[ "$observed_parent" == "$$" && \
                  "$observed_start" =~ ^[0-9]+$ ]]; then
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

terminate_exact_fork() {
    local pid=$1
    local expected_start=$2
    local observed_parent observed_start
    if [[ -r "/proc/$pid/status" && -r "/proc/$pid/stat" ]]; then
        observed_parent=$(awk '/^PPid:/{print $2}' "/proc/$pid/status")
        observed_start=$(proc_start "$pid" || true)
        if [[ "$observed_parent" == "$$" && \
              "$observed_start" == "$expected_start" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
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
    if ") " not in stat_line:
        raise SystemExit(f"child {pid} has malformed stat identity")
    stat_fields = stat_line.rsplit(") ", 1)[1].split()
    argv = [token for token in cmdline.split(b"\0") if token]
    if (
        stat_fields[0] != "Z"
        and int(stat_fields[1]) == expected_ppid
        and argv == expected_argv
    ):
        print(stat_fields[19])
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
        observed_start=$(proc_start "$pid")
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

launch_registered_gpu_child() {
    local log_path=$1
    shift
    local -a command=("$@")
    local fork_start
    "${command[@]}" >"$log_path" 2>&1 &
    local pid=$!
    fork_start=$(capture_fork_start "$pid" || true)
    if [[ ! "$fork_start" =~ ^[0-9]+$ ]] || \
       ! register_pid "$pid" "${command[@]}"; then
        if [[ "$fork_start" =~ ^[0-9]+$ ]]; then
            terminate_exact_fork "$pid" "$fork_start"
        else
            wait "$pid" 2>/dev/null || true
        fi
        printf 'failed to register exact GPU child: %s\n' "$pid" >&2
        return 1
    fi
    LAST_CHILD_PID=$pid
}

run_registered_gpu_child() {
    local log_path=$1
    shift
    local -a command=("$@")
    launch_registered_gpu_child "$log_path" "${command[@]}"
    if ! wait "$LAST_CHILD_PID"; then
        printf 'registered GPU child failed: %s\n' "${command[*]}" >&2
        return 1
    fi
}

mapfile -d '' -t epochs < <(
    "$python_bin" - "$partition_id" "$partition_count" <<'PY'
from scripts.show_base import base_fresh_val_orchestrator as task
import sys

for epoch in task._partition_epochs(int(sys.argv[1]), int(sys.argv[2])):
    sys.stdout.write(str(epoch))
    sys.stdout.write("\0")
PY
)
if [[ ${#epochs[@]} -lt 1 ]]; then
    printf 'static partition is empty\n' >&2
    exit 1
fi

seal_paths=()
seal_shas=()
seal_payloads=()
declare -A lineage_by_epoch lineage_sha_by_epoch manifest_by_epoch
declare -A manifest_sha_by_epoch manifest_bytes_by_epoch
declare -A gate_path_by_epoch gate_sha_by_epoch gate_payload_by_epoch
declare -A distribution_by_epoch distribution_sha_by_epoch
declare -A distribution_bytes_by_epoch distribution_payload_by_epoch
declare -A transaction_by_epoch transaction_sha_by_epoch
declare -A transaction_payload_by_epoch metric_gpu_by_epoch
declare -A screen_by_epoch screen_sha_by_epoch screen_bytes_by_epoch
declare -A screen_payload_by_epoch replay_by_epoch replay_sha_by_epoch
declare -A replay_payload_by_epoch

for ((wave_start = 0; wave_start < ${#epochs[@]}; \
      wave_start += candidates_per_wave)); do
    wave_stop=$((wave_start + candidates_per_wave))
    if ((wave_stop > ${#epochs[@]})); then
        wave_stop=${#epochs[@]}
    fi
    wave_epochs=("${epochs[@]:wave_start:wave_stop-wave_start}")
    shard_pids=()
    for epoch in "${wave_epochs[@]}"; do
        candidate_root="$run_root/candidates/e$epoch"
        mkdir "$candidate_root"
        for shard_id in 0 1 2 3 4 5 6 7; do
            shard_command=(
                "$python_bin" "$inference" shard --split val
                --preflight "$preflight"
                --expected-preflight-sha256 "$preflight_sha"
                --epoch "$epoch" --output-root "$candidate_root"
                --num-shards 8 --shard-id "$shard_id"
                --device "cuda:$shard_id" --seed "$seed"
            )
            launch_registered_gpu_child \
                "$run_root/logs/e${epoch}-shard${shard_id}.log" \
                "${shard_command[@]}"
            shard_pids+=("$LAST_CHILD_PID")
        done
    done
    shard_failure=0
    for pid in "${shard_pids[@]}"; do
        if ! wait "$pid"; then
            shard_failure=1
        fi
    done
    if [[ "$shard_failure" -ne 0 ]]; then
        printf 'Base validation wave failed; no complete receipt will be published: %s\n' \
            "${wave_epochs[*]}" >&2
        exit 1
    fi

    screen_pids=()
    for ((wave_offset = 0; wave_offset < ${#wave_epochs[@]}; \
          wave_offset++)); do
        epoch=${wave_epochs[$wave_offset]}
        candidate_root="$run_root/candidates/e$epoch"
        metric_gpu=$((wave_offset * 8 / ${#wave_epochs[@]}))
        "$python_bin" "$inference" finalize --split val \
            --preflight "$preflight" \
            --expected-preflight-sha256 "$preflight_sha" \
            --epoch "$epoch" --output-root "$candidate_root" \
            --num-shards 8 \
            >"$run_root/logs/e${epoch}-finalize.log" 2>&1
        lineage="$candidate_root/final/val-inference-lineage.json"
        manifest="$candidate_root/final/final_manifest.jsonl"
        artifact_fields "$lineage" true
        lineage_sha=${artifact_result[0]}
        lineage_payload=${artifact_result[2]}
        artifact_fields "$manifest"
        manifest_sha=${artifact_result[0]}
        manifest_bytes=${artifact_result[1]}

        mapfile -d '' -t candidate_context < <(
            "$python_bin" "$orchestrator" extract-candidate-context \
            --run-spec-path "$run_spec" \
            --run-spec-sha256 "$run_spec_sha256" \
            --run-spec-payload-sha256 "$run_spec_payload_sha256" \
            --source-commit "$source_commit" --source-tree "$source_tree" \
            --preflight-path "$preflight" \
            --preflight-sha256 "$preflight_sha" \
            --preflight-payload-sha256 "$preflight_payload" \
            --epoch "$epoch"
        )
        if [[ ${#candidate_context[@]} -ne 6 ]]; then
            printf 'candidate gate/checkpoint extraction failed\n' >&2
            exit 1
        fi
        gate_path=${candidate_context[0]}
        gate_sha=${candidate_context[1]}
        gate_payload=${candidate_context[2]}
        checkpoint=(
            "${candidate_context[3]}"
            "${candidate_context[4]}"
            "${candidate_context[5]}"
        )

        distribution="$candidate_root/distribution.json"
        "$python_bin" "$orchestrator" distribution \
        --lineage-path "$lineage" --lineage-sha256 "$lineage_sha" \
        --lineage-payload-sha256 "$lineage_payload" \
        --gate-path "$gate_path" --gate-sha256 "$gate_sha" \
        --checkpoint-path "${checkpoint[0]}" \
        --checkpoint-sha256 "${checkpoint[1]}" \
        --checkpoint-bytes "${checkpoint[2]}" \
        --output-json "$distribution" \
            >"$run_root/logs/e${epoch}-distribution.log" 2>&1
        artifact_fields "$distribution" true
        distribution_sha=${artifact_result[0]}
        distribution_bytes=${artifact_result[1]}
        distribution_payload=${artifact_result[2]}

        failure="$candidate_root/failure-manifest.json"
        "$python_bin" "$orchestrator" failure-manifest --epoch "$epoch" \
        --output-json "$failure" \
        >"$run_root/logs/e${epoch}-failure-manifest.log" 2>&1
        artifact_fields "$failure" true
        failure_sha=${artifact_result[0]}
        failure_payload=${artifact_result[2]}

        transaction="$candidate_root/candidate-transaction.json"
        "$python_bin" "$orchestrator" candidate-transaction --epoch "$epoch" \
        --formal-host "$formal_host" \
        --output-root "$candidate_root" \
        --preflight-path "$preflight" --preflight-sha256 "$preflight_sha" \
        --preflight-payload-sha256 "$preflight_payload" \
        --lineage-path "$lineage" --lineage-sha256 "$lineage_sha" \
        --lineage-payload-sha256 "$lineage_payload" \
        --distribution-path "$distribution" \
        --distribution-sha256 "$distribution_sha" \
        --distribution-payload-sha256 "$distribution_payload" \
        --failure-path "$failure" --failure-sha256 "$failure_sha" \
        --failure-payload-sha256 "$failure_payload" \
        --prerequisite-path "$prerequisite" \
        --prerequisite-sha256 "$prerequisite_sha" \
        --prerequisite-payload-sha256 "$prerequisite_payload" \
        --continuation-path "$continuation" \
        --continuation-sha256 "$continuation_sha" \
        --continuation-payload-sha256 "$continuation_payload" \
        --output-json "$transaction" \
            >"$run_root/logs/e${epoch}-transaction.log" 2>&1
        artifact_fields "$transaction" true
        transaction_sha=${artifact_result[0]}
        transaction_payload=${artifact_result[2]}

        lineage_by_epoch[$epoch]=$lineage
        lineage_sha_by_epoch[$epoch]=$lineage_sha
        manifest_by_epoch[$epoch]=$manifest
        manifest_sha_by_epoch[$epoch]=$manifest_sha
        manifest_bytes_by_epoch[$epoch]=$manifest_bytes
        gate_path_by_epoch[$epoch]=$gate_path
        gate_sha_by_epoch[$epoch]=$gate_sha
        gate_payload_by_epoch[$epoch]=$gate_payload
        distribution_by_epoch[$epoch]=$distribution
        distribution_sha_by_epoch[$epoch]=$distribution_sha
        distribution_bytes_by_epoch[$epoch]=$distribution_bytes
        distribution_payload_by_epoch[$epoch]=$distribution_payload
        transaction_by_epoch[$epoch]=$transaction
        transaction_sha_by_epoch[$epoch]=$transaction_sha
        transaction_payload_by_epoch[$epoch]=$transaction_payload
        metric_gpu_by_epoch[$epoch]=$metric_gpu
        screen="$candidate_root/released2-primary-screen.json"
        screen_by_epoch[$epoch]=$screen
        screen_command=(
            "$python_bin" "$replay" screen
            --talkshow-metric-root "$metric_root"
            --feature-extractor "$feature_extractor"
            --smplx-asset "$smplx_asset"
            --device "cuda:$metric_gpu"
            --split val --expected-clip-count 1715
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
        launch_registered_gpu_child \
            "$run_root/logs/e${epoch}-primary-screen.log" \
            "${screen_command[@]}"
        screen_pids+=("$LAST_CHILD_PID")
    done

    screen_failure=0
    for pid in "${screen_pids[@]}"; do
        if ! wait "$pid"; then
            screen_failure=1
        fi
    done
    if [[ "$screen_failure" -ne 0 ]]; then
        printf 'Base validation primary-screen wave failed: %s\n' \
            "${wave_epochs[*]}" >&2
        exit 1
    fi

    # The first-pass screen receipt is not allowed to rank itself.  A second
    # feature pass independently reopens every byte-pinned raw prediction NPZ
    # and must reproduce released2 FGD exactly before this row can be sealed.
    replay_pids=()
    for epoch in "${wave_epochs[@]}"; do
        screen=${screen_by_epoch[$epoch]}
        artifact_fields "$screen" true
        screen_sha_by_epoch[$epoch]=${artifact_result[0]}
        screen_bytes_by_epoch[$epoch]=${artifact_result[1]}
        screen_payload_by_epoch[$epoch]=${artifact_result[2]}
        replay_receipt=
        replay_receipt+="$run_root/candidates/e$epoch/"
        replay_receipt+="released2-primary-replay.json"
        replay_by_epoch[$epoch]=$replay_receipt
        replay_command=(
            "$python_bin" "$replay" replay
            --talkshow-metric-root "$metric_root"
            --feature-extractor "$feature_extractor"
            --smplx-asset "$smplx_asset"
            --device "cuda:${metric_gpu_by_epoch[$epoch]}"
            --split val --expected-clip-count 1715
            --report-json "$screen"
            --expected-report-sha256 "${screen_sha_by_epoch[$epoch]}"
            --expected-report-bytes "${screen_bytes_by_epoch[$epoch]}"
            --prediction-manifest "${manifest_by_epoch[$epoch]}"
            --expected-prediction-manifest-sha256 \
                "${manifest_sha_by_epoch[$epoch]}"
            --expected-prediction-manifest-bytes \
                "${manifest_bytes_by_epoch[$epoch]}"
            --real-feature-cache-json "$real_cache"
            --expected-real-feature-cache-sha256 "$real_cache_sha"
            --expected-real-feature-cache-bytes "$real_cache_bytes"
            --expected-real-feature-cache-payload-sha256 \
                "$real_cache_payload"
            --output-json "$replay_receipt"
        )
        launch_registered_gpu_child \
            "$run_root/logs/e${epoch}-primary-replay.log" \
            "${replay_command[@]}"
        replay_pids+=("$LAST_CHILD_PID")
    done

    replay_failure=0
    for pid in "${replay_pids[@]}"; do
        if ! wait "$pid"; then
            replay_failure=1
        fi
    done
    if [[ "$replay_failure" -ne 0 ]]; then
        printf 'Base validation raw-prediction replay wave failed: %s\n' \
            "${wave_epochs[*]}" >&2
        exit 1
    fi

    for epoch in "${wave_epochs[@]}"; do
        replay_receipt=${replay_by_epoch[$epoch]}
        artifact_fields "$replay_receipt" true
        replay_sha_by_epoch[$epoch]=${artifact_result[0]}
        replay_payload_by_epoch[$epoch]=${artifact_result[2]}
        seal="$run_root/seals/e${epoch}.json"
        "$python_bin" "$orchestrator" candidate-seal \
            --transaction-path "${transaction_by_epoch[$epoch]}" \
            --transaction-sha256 "${transaction_sha_by_epoch[$epoch]}" \
            --transaction-payload-sha256 "${transaction_payload_by_epoch[$epoch]}" \
            --primary-screen-path "$screen" \
            --primary-screen-sha256 "${screen_sha_by_epoch[$epoch]}" \
            --primary-screen-payload-sha256 "${screen_payload_by_epoch[$epoch]}" \
            --primary-replay-path "$replay_receipt" \
            --primary-replay-sha256 "${replay_sha_by_epoch[$epoch]}" \
            --primary-replay-payload-sha256 \
                "${replay_payload_by_epoch[$epoch]}" \
            --prerequisite-path "$prerequisite" \
            --prerequisite-sha256 "$prerequisite_sha" \
            --prerequisite-payload-sha256 "$prerequisite_payload" \
            --continuation-path "$continuation" \
            --continuation-sha256 "$continuation_sha" \
            --continuation-payload-sha256 "$continuation_payload" \
            --real-feature-cache-path "$real_cache" \
            --real-feature-cache-sha256 "$real_cache_sha" \
            --real-feature-cache-payload-sha256 "$real_cache_payload" \
            --output-json "$seal" \
            >"$run_root/logs/e${epoch}-seal.log" 2>&1
        artifact_fields "$seal" true
        seal_paths+=("$seal")
        seal_shas+=("${artifact_result[0]}")
        seal_payloads+=("${artifact_result[2]}")
    done
done

partition_receipt="$run_root/partition-receipt.json"
partition_args=(
    "$python_bin" "$orchestrator" seal-partition
    --partition-id "$partition_id" --partition-count "$partition_count"
)
for ((index = 0; index < ${#seal_paths[@]}; index++)); do
    partition_args+=(
        --candidate-seal "${seal_paths[$index]}"
        --candidate-seal-sha256 "${seal_shas[$index]}"
        --candidate-seal-payload-sha256 "${seal_payloads[$index]}"
    )
done
partition_args+=(--output-json "$partition_receipt")
"${partition_args[@]}" >"$run_root/logs/partition-seal.log" 2>&1

trap - EXIT INT TERM
artifact_fields "$partition_receipt" true
printf '%s\0%s\0%s\0' \
    "$partition_receipt" "${artifact_result[0]}" "${artifact_result[2]}"
