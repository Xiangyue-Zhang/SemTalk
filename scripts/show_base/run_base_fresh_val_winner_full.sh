#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Stage B: direct child only of the all-GPU guarded runner.  It accepts an
# immutable primary-screen winner, freshly replays that winner against the
# frozen run specification, then computes exactly one full validation report
# and one released2 replay.  It never reads or writes a test artifact.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON RUN_SPEC RUN_SPEC_SHA256 RUN_SPEC_PAYLOAD_SHA256 WINNER WINNER_SHA256 WINNER_PAYLOAD_SHA256 RUN_ROOT SOURCE_COMMIT SOURCE_TREE"
}

if [[ $# -ne 11 ]]; then
    usage >&2
    exit 2
fi

repo_root=$1
python_bin=$2
run_spec=$3
run_spec_sha256=$4
run_spec_payload_sha256=$5
winner=$6
winner_sha256=$7
winner_payload_sha256=$8
run_root=$9
source_commit=${10}
source_tree=${11}

if [[ ! "$repo_root" = /* || ! "$python_bin" = /* || \
      ! "$run_spec" = /* || ! "$winner" = /* || ! "$run_root" = /* || \
      ! -x "$python_bin" || ! -f "$run_spec" || -L "$run_spec" || \
      ! -f "$winner" || -L "$winner" || \
      -e "$run_root" || -L "$run_root" || \
      ! -d "$(dirname "$run_root")" ]]; then
    printf 'winner-full inputs must be canonical create-new paths\n' >&2
    exit 2
fi
for digest in \
    "$run_spec_sha256" "$run_spec_payload_sha256" \
    "$winner_sha256" "$winner_payload_sha256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
for oid in "$source_commit" "$source_tree"; do
    [[ "$oid" =~ ^[0-9a-f]{40}$ ]] || exit 2
done

orchestrator="$repo_root/scripts/show_base/base_fresh_val_orchestrator.py"
metrics="$repo_root/scripts/show_base/evaluate_talkshow_show_metrics.py"
replay="$repo_root/scripts/show_base/replay_released2_primary.py"
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
for required in "$orchestrator" "$metrics" "$replay"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        printf 'required winner-full source is unavailable: %s\n' \
            "$required" >&2
        exit 1
    fi
done
if [[ ! -f "$launcher_dir/guarded_runner_contract.sh" || \
      -L "$launcher_dir/guarded_runner_contract.sh" ]]; then
    printf 'guarded runner contract is unavailable\n' >&2
    exit 1
fi
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

if [[ "$(git -C "$repo_root" remote get-url origin)" != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      "$(git -C "$repo_root" rev-parse HEAD)" != "$source_commit" || \
      "$(git -C "$repo_root" rev-parse 'HEAD^{tree}')" != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'winner-full source must be exact clean detached SemTalk with zero branches\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/guarded_runner_contract.sh \
    scripts/show_base/base_fresh_val_orchestrator.py \
    scripts/show_base/evaluate_talkshow_show_metrics.py \
    scripts/show_base/replay_released2_primary.py \
    scripts/show_base/published_test_winner_claim.py \
    scripts/show_base/select_published_base_winner.py; do
    if [[ "$(git -C "$repo_root" ls-files --error-unmatch "$tracked")" != \
          "$tracked" ]]; then
        printf 'winner-full source is not tracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

mapfile -d '' -t context < <(
    "$python_bin" "$orchestrator" extract-winner-full-context \
        --run-spec-path "$run_spec" \
        --run-spec-sha256 "$run_spec_sha256" \
        --run-spec-payload-sha256 "$run_spec_payload_sha256" \
        --winner-selection-path "$winner" \
        --winner-selection-sha256 "$winner_sha256" \
        --winner-selection-payload-sha256 "$winner_payload_sha256" \
        --source-commit "$source_commit" --source-tree "$source_tree"
)
if [[ ${#context[@]} -ne 36 ]]; then
    printf 'winner-full context extraction failed\n' >&2
    exit 1
fi
epoch=${context[0]}
updates=${context[1]}
prediction=${context[2]}
prediction_sha=${context[3]}
prediction_bytes=${context[4]}
lineage=${context[5]}
lineage_sha=${context[6]}
lineage_bytes=${context[7]}
lineage_payload=${context[8]}
distribution=${context[9]}
distribution_sha=${context[10]}
distribution_bytes=${context[11]}
distribution_payload=${context[12]}
gate=${context[13]}
gate_sha=${context[14]}
gate_bytes=${context[15]}
gate_payload=${context[16]}
canonical=${context[17]}
canonical_sha=${context[18]}
canonical_bytes=${context[19]}
metric_root=${context[20]}
feature_extractor=${context[21]}
smplx_asset=${context[22]}
real_cache=${context[23]}
real_cache_sha=${context[24]}
real_cache_bytes=${context[25]}
real_cache_payload=${context[26]}
transaction=${context[27]}
transaction_sha=${context[28]}
transaction_bytes=${context[29]}
transaction_payload=${context[30]}
screen=${context[31]}
screen_sha=${context[32]}
screen_bytes=${context[33]}
screen_payload=${context[34]}
expected_formal_host=${context[35]}
if [[ "$(hostname)" != "$expected_formal_host" ]]; then
    printf 'winner-full requires exact selected formal host %s, observed %s\n' \
        "$expected_formal_host" "$(hostname)" >&2
    exit 1
fi

mapfile -d '' -t run_root_identity < <(
    "$python_bin" "$orchestrator" create-run-root --path "$run_root" \
        --subdirectory logs
)
if [[ ${#run_root_identity[@]} -ne 3 || \
      "${run_root_identity[0]}" != "$run_root" || \
      ! ${run_root_identity[1]} =~ ^[0-9]+$ || \
      ! ${run_root_identity[2]} =~ ^[0-9]+$ ]]; then
    printf 'winner-full run root could not be created safely\n' >&2
    exit 1
fi
"$python_bin" "$orchestrator" validate-run-spec \
    --run-spec-path "$run_spec" \
    --run-spec-sha256 "$run_spec_sha256" \
    --run-spec-payload-sha256 "$run_spec_payload_sha256" \
    --source-commit "$source_commit" --source-tree "$source_tree" \
    --output-json "$run_root/run-spec-audit.json" \
    >"$run_root/logs/run-spec.log" 2>&1

mapfile -d '' -t common < <(
    "$python_bin" "$orchestrator" extract-run-spec \
        --run-spec-path "$run_spec" \
        --run-spec-sha256 "$run_spec_sha256" \
        --run-spec-payload-sha256 "$run_spec_payload_sha256" \
        --source-commit "$source_commit" --source-tree "$source_tree" \
        --profile finalizer
)
if [[ ${#common[@]} -lt 12 || $(((${#common[@]} - 12) % 4)) -ne 0 ]]; then
    printf 'run-spec finalizer extraction failed\n' >&2
    exit 1
fi
prerequisite=${common[0]}
prerequisite_sha=${common[1]}
prerequisite_bytes=${common[2]}
prerequisite_payload=${common[3]}
continuation=${common[4]}
continuation_sha=${common[5]}
continuation_bytes=${common[6]}
continuation_payload=${common[7]}
if [[ "$real_cache" != "${common[8]}" || \
      "$real_cache_sha" != "${common[9]}" || \
      "$real_cache_bytes" != "${common[10]}" || \
      "$real_cache_payload" != "${common[11]}" ]]; then
    printf 'winner-full cache differs from run specification\n' >&2
    exit 1
fi

artifact_fields() {
    local path=$1
    local payload_flag=${2:-false}
    local -a command=(
        "$python_bin" "$orchestrator" artifact-fields
        --artifact-path "$path"
    )
    if [[ "$payload_flag" == true ]]; then
        command+=(--payload-receipt)
    fi
    mapfile -d '' -t artifact_result < <("${command[@]}")
    if [[ ${#artifact_result[@]} -ne 3 || \
          ! ${artifact_result[0]} =~ ^[0-9a-f]{64}$ ]]; then
        printf 'cannot pin winner-full artifact: %s\n' "$path" >&2
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
        printf 'winner-full child identity/parent/full argv mismatch: %s\n' \
            "$pid" >&2
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

run_registered_gpu_child() {
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
        printf 'failed to register exact winner-full GPU child: %s\n' \
            "$pid" >&2
        return 1
    fi
    if ! wait "$pid"; then
        printf 'registered winner-full GPU child failed: %s\n' \
            "${command[*]}" >&2
        return 1
    fi
}

report="$run_root/winner-e${epoch}-talkshow-metrics.json"
metric_command=(
    "$python_bin" "$metrics"
    --canonical-manifest "$canonical"
    --expected-canonical-manifest-sha256 "$canonical_sha"
    --prediction-manifest "$prediction"
    --expected-prediction-manifest-sha256 "$prediction_sha"
    --prediction-lineage "$lineage"
    --expected-prediction-lineage-sha256 "$lineage_sha"
    --validation-gate-json "$gate"
    --expected-validation-gate-sha256 "$gate_sha"
    --expected-validation-gate-receipt-payload-sha256 "$gate_payload"
    --distribution-declaration-json "$distribution"
    --expected-distribution-declaration-sha256 "$distribution_sha"
    --talkshow-metric-root "$metric_root"
    --feature-extractor "$feature_extractor"
    --smplx-asset "$smplx_asset"
    --device cuda:0 --split val --expected-clip-count 1715
    --torch-threads 1 --output-json "$report"
)
run_registered_gpu_child "$run_root/logs/full-metrics.log" \
    "${metric_command[@]}"
artifact_fields "$report"
report_sha=${artifact_result[0]}
report_bytes=${artifact_result[1]}

primary_replay="$run_root/winner-e${epoch}-released2-primary-replay.json"
replay_command=(
    "$python_bin" "$replay" replay
    --talkshow-metric-root "$metric_root"
    --feature-extractor "$feature_extractor"
    --smplx-asset "$smplx_asset"
    --device cuda:0 --split val --expected-clip-count 1715
    --torch-threads 1
    --report-json "$report"
    --expected-report-sha256 "$report_sha"
    --expected-report-bytes "$report_bytes"
    --prediction-manifest "$prediction"
    --expected-prediction-manifest-sha256 "$prediction_sha"
    --expected-prediction-manifest-bytes "$prediction_bytes"
    --real-feature-cache-json "$real_cache"
    --expected-real-feature-cache-sha256 "$real_cache_sha"
    --expected-real-feature-cache-bytes "$real_cache_bytes"
    --expected-real-feature-cache-payload-sha256 "$real_cache_payload"
    --output-json "$primary_replay"
)
run_registered_gpu_child "$run_root/logs/primary-replay.log" \
    "${replay_command[@]}"
artifact_fields "$primary_replay" true
replay_sha=${artifact_result[0]}
replay_payload=${artifact_result[2]}

closure="$run_root/winner-full-metric-closure.json"
"$python_bin" "$orchestrator" winner-full-closure \
    --winner-selection-path "$winner" \
    --winner-selection-sha256 "$winner_sha256" \
    --winner-selection-payload-sha256 "$winner_payload_sha256" \
    --prerequisite-path "$prerequisite" \
    --prerequisite-sha256 "$prerequisite_sha" \
    --prerequisite-payload-sha256 "$prerequisite_payload" \
    --continuation-path "$continuation" \
    --continuation-sha256 "$continuation_sha" \
    --continuation-payload-sha256 "$continuation_payload" \
    --real-feature-cache-path "$real_cache" \
    --real-feature-cache-sha256 "$real_cache_sha" \
    --real-feature-cache-payload-sha256 "$real_cache_payload" \
    --primary-replay-path "$primary_replay" \
    --primary-replay-sha256 "$replay_sha" \
    --primary-replay-payload-sha256 "$replay_payload" \
    --report-path "$report" --report-sha256 "$report_sha" \
    --output-json "$closure" >"$run_root/logs/closure.log" 2>&1
artifact_fields "$closure" true

# The guarded parent performs guard restoration after this direct child exits.
trap - EXIT INT TERM
printf '%s\0%s\0%s\0' \
    "$closure" "${artifact_result[0]}" "${artifact_result[2]}"
