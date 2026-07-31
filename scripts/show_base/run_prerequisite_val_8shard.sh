#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# This launcher is valid only as the direct workload of
# /tmp/globaldiff_guarded_runner.py --gpus 0,1,2,3,4,5,6,7.  For each of the
# independent stage/candidate measurements it launches exactly eight modulo
# shards, one per physical GPU.  Up to four candidates may share the eight
# GPUs concurrently (four evaluator processes per GPU); every candidate keeps
# the unchanged eight-shard numeric protocol and merge order.  No test data
# and no complete-Base FGD enter this protocol.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON CANDIDATE_INDEX CANDIDATE_INDEX_SHA256 CANONICAL_MANIFEST MANIFEST_SHA256 CANONICAL_SUMMARY SUMMARY_SHA256 CANONICAL_LINEAGE LINEAGE_SHA256 SHARD_ROOT MEASUREMENT_ROOT SELECTION_JSON RUN_ID SOURCE_COMMIT SOURCE_TREE"
}

if [[ $# -ne 16 ]]; then
    usage >&2
    exit 2
fi

launcher_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=guarded_runner_contract.sh
. "$launcher_dir/guarded_runner_contract.sh"
# shellcheck source=prerequisite_val_launcher_contract.sh
. "$launcher_dir/prerequisite_val_launcher_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

repo_root=$1
python_bin=$2
candidate_index=$3
candidate_index_sha256=$4
canonical_manifest=$5
manifest_sha256=$6
canonical_summary=$7
summary_sha256=$8
canonical_lineage=$9
lineage_sha256=${10}
shard_root=${11}
measurement_root=${12}
selection_json=${13}
run_id=${14}
source_commit=${15}
source_tree=${16}
batch_size=${SEMTALK_PREREQ_VAL_BATCH_SIZE:-32}
partition=${SEMTALK_PREREQ_VAL_PARTITION:-all}
candidates_per_wave=${SEMTALK_PREREQ_VAL_CANDIDATES_PER_WAVE:-4}
gate_receipt=${SEMTALK_PREREQ_VAL_MULTICANDIDATE_GATE_RECEIPT:-}
gate_receipt_sha256=${SEMTALK_PREREQ_VAL_MULTICANDIDATE_GATE_SHA256:-}

case "$partition" in
    all)
        partition_modulus=1
        partition_remainder=0
        ;;
    0of2)
        partition_modulus=2
        partition_remainder=0
        ;;
    1of2)
        partition_modulus=2
        partition_remainder=1
        ;;
    nonglobal|global|gate)
        partition_modulus=1
        partition_remainder=0
        ;;
    *)
        printf '%s\n' \
            "SEMTALK_PREREQ_VAL_PARTITION must be all, 0of2, 1of2, nonglobal, global, or gate" >&2
        exit 2
        ;;
esac

if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'unsafe run id: %s\n' "$run_id" >&2
    exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    printf 'SEMTALK_PREREQ_VAL_BATCH_SIZE must be positive\n' >&2
    exit 2
fi
if [[ ! "$candidates_per_wave" =~ ^[1-4]$ ]]; then
    printf 'SEMTALK_PREREQ_VAL_CANDIDATES_PER_WAVE must be 1..4\n' >&2
    exit 2
fi
for oid in "$source_commit" "$source_tree"; do
    if [[ ! "$oid" =~ ^[0-9a-f]{40}$ ]]; then
        printf 'invalid source Git object ID\n' >&2
        exit 2
    fi
done
for digest in "$candidate_index_sha256" "$manifest_sha256" \
    "$summary_sha256" "$lineage_sha256"; do
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'invalid frozen SHA-256 argument\n' >&2
        exit 2
    fi
done

evaluator="$repo_root/scripts/show_base/evaluate_prerequisite_val_shard.py"
merger="$repo_root/scripts/show_base/merge_prerequisite_val_shards.py"
selector="$repo_root/scripts/show_base/select_prerequisite_candidates.py"
gate_checker="$repo_root/scripts/show_base/check_prerequisite_val_multicandidate_gate.py"
for required in "$python_bin" "$candidate_index" "$canonical_manifest" \
    "$canonical_summary" "$canonical_lineage" "$evaluator" "$merger" \
    "$selector" "$gate_checker"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        printf 'required regular input is missing: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ ! -x "$python_bin" ]]; then
    printf 'Python executable is not executable: %s\n' "$python_bin" >&2
    exit 1
fi
if [[ ! "$shard_root" = /* || ! "$measurement_root" = /* || \
      ! "$selection_json" = /* ]]; then
    printf 'all output paths must be absolute\n' >&2
    exit 2
fi
if [[ -e "$shard_root" || -L "$shard_root" || \
      -e "$measurement_root" || -L "$measurement_root" || \
      -e "$selection_json" || -L "$selection_json" ]]; then
    printf 'refusing to reuse any formal validation output\n' >&2
    exit 1
fi
if [[ ! -d "$(dirname "$shard_root")" || \
      ! -d "$(dirname "$measurement_root")" || \
      ! -d "$(dirname "$selection_json")" ]]; then
    printf 'all output parents must already exist\n' >&2
    exit 1
fi

# Fail closed unless the unique guarded runner owns all eight physical GPUs.
runner_pid=$PPID
if [[ ! -r "/proc/$runner_pid/stat" || \
      ! -r "/proc/$runner_pid/cmdline" ]]; then
    printf 'guarded runner process is unavailable: %s\n' "$runner_pid" >&2
    exit 1
fi
mapfile -d '' -t runner_argv <"/proc/$runner_pid/cmdline"
runner_path_count=0
runner_gpus_count=0
for ((index = 0; index < ${#runner_argv[@]}; index++)); do
    if [[ ${runner_argv[$index]} == /tmp/globaldiff_guarded_runner.py ]]; then
        ((runner_path_count += 1))
    fi
    if [[ ${runner_argv[$index]} == --gpus && \
          $((index + 1)) -lt ${#runner_argv[@]} && \
          ${runner_argv[$((index + 1))]} == 0,1,2,3,4,5,6,7 ]]; then
        ((runner_gpus_count += 1))
    fi
    if [[ ${runner_argv[$index]} == --gpus=0,1,2,3,4,5,6,7 ]]; then
        ((runner_gpus_count += 1))
    fi
done
if [[ "$runner_path_count" -ne 1 || "$runner_gpus_count" -ne 1 ]]; then
    printf 'requires one exact guarded-runner all-GPU reservation\n' >&2
    exit 1
fi

if [[ "$(git -C "$repo_root" remote get-url origin)" != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      "$(git -C "$repo_root" rev-parse HEAD)" != "$source_commit" || \
      "$(git -C "$repo_root" rev-parse 'HEAD^{tree}')" != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'formal source must be exact, clean, detached SemTalk with zero local branches\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/evaluate_prerequisite_val_shard.py \
    scripts/show_base/check_prerequisite_val_multicandidate_gate.py \
    scripts/show_base/merge_prerequisite_val_shards.py \
    scripts/show_base/select_prerequisite_candidates.py; do
    if [[ "$(git -C "$repo_root" ls-files --error-unmatch "$tracked")" != \
          "$tracked" ]]; then
        printf 'formal source is untracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

if [[ "$partition" != gate && "$candidates_per_wave" -gt 1 ]]; then
    if [[ ! "$gate_receipt" = /* || \
          ! "$gate_receipt_sha256" =~ ^[0-9a-f]{64}$ || \
          ! -f "$gate_receipt" || -L "$gate_receipt" ]]; then
        printf 'multi-candidate formal run requires an absolute frozen gate receipt and SHA-256\n' >&2
        exit 1
    fi
    "$python_bin" - "$gate_receipt" "$gate_receipt_sha256" \
        "$candidate_index" "$candidate_index_sha256" "$manifest_sha256" \
        "$summary_sha256" "$lineage_sha256" "$source_commit" "$source_tree" \
        "$batch_size" <<'PY'
from pathlib import Path
import sys

from scripts.show_base import check_prerequisite_val_multicandidate_gate as gate

receipt = gate.replay_gate(Path(sys.argv[1]), sys.argv[2])
common = receipt["inputs"]["common"]
expected = {
    "candidate_index": str(Path(sys.argv[3]).resolve(strict=True)),
    "candidate_index_sha256": sys.argv[4],
    "canonical_manifest_sha256": sys.argv[5],
    "canonical_summary_sha256": sys.argv[6],
    "canonical_lineage_sha256": sys.argv[7],
    "source_commit": sys.argv[8],
    "source_tree": sys.argv[9],
    "multi_candidate_gate": None,
}
if common != expected:
    raise SystemExit("multi-candidate gate input binding mismatch")
serial = receipt["inputs"]["serial_partition"]
concurrent = receipt["inputs"]["concurrent_partition"]
if (
    receipt["protocol"]["concurrent_candidates_per_wave"] != 4
    or receipt["protocol"]["shards_per_candidate"] != 8
    or receipt["protocol"]["batch_size"] != int(sys.argv[10])
    or receipt["throughput"]["pass"] is not True
):
    raise SystemExit("multi-candidate gate protocol mismatch")
for binding in (serial, concurrent):
    if set(binding) != {"path", "sha256", "receipt_payload_sha256"}:
        raise SystemExit("multi-candidate gate binding schema mismatch")
PY
fi

mkdir "$shard_root"
mkdir "$shard_root/logs" "$shard_root/shards"
plan_path="$shard_root/candidate_plan.tsv"

cd "$repo_root"
"$python_bin" - "$candidate_index" "$candidate_index_sha256" \
    "$canonical_manifest" "$manifest_sha256" "$canonical_summary" \
    "$summary_sha256" "$canonical_lineage" "$lineage_sha256" \
    "$plan_path" <<'PY'
from pathlib import Path
import sys

from scripts.show_base import prerequisite_val_contract as contract

index, _ = contract.load_candidate_index(
    Path(sys.argv[1]),
    sys.argv[2],
    allow_partial=True,
)
rows, _ = contract.load_val_canonical(
    manifest_path=Path(sys.argv[3]),
    manifest_sha256=sys.argv[4],
    summary_path=Path(sys.argv[5]),
    summary_sha256=sys.argv[6],
    lineage_path=Path(sys.argv[7]),
    lineage_sha256=sys.argv[8],
)
if len(rows) != contract.EXPECTED_VAL_CLIPS:
    raise SystemExit("canonical validation coverage mismatch")
lines = []
index_stages = tuple(
    stage for stage in contract.STAGES if stage in index["stages"]
)
is_partial = (
    index["format"] in {
        contract.PARTIAL_CANDIDATE_INDEX_FORMAT,
        contract.SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT,
    }
)
if set(index_stages) not in (
    set(contract.STAGES),
    set(contract.STAGES[:-1]),
):
    raise SystemExit("candidate index stage authority is invalid")
for stage in index_stages:
    for epoch in contract.candidate_epochs_for_stage(index, stage):
        item = contract.candidate_lookup(index, stage, epoch)
        fields = (
            stage,
            str(epoch),
            str(item["optimizer_updates"]),
            item["checkpoint"],
            item["checkpoint_sha256"],
        )
        if any("\t" in field or "\n" in field for field in fields):
            raise SystemExit("candidate plan contains unsafe text")
        lines.append("\t".join(fields))
expected = sum(
    len(contract.candidate_epochs_for_stage(index, stage))
    for stage in index_stages
)
if len(lines) != expected:
    raise SystemExit("candidate plan is not exact")
path = Path(sys.argv[9])
with path.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(lines) + "\n")
PY

mapfile -t candidate_index_formats < <(
    "$python_bin" - "$candidate_index" "$candidate_index_sha256" <<'PY'
from pathlib import Path
import sys
from scripts.show_base import prerequisite_val_contract as contract
value, _ = contract.load_candidate_index(
    Path(sys.argv[1]), sys.argv[2], allow_partial=True
)
print(value["format"])
print(contract.PARTIAL_CANDIDATE_INDEX_FORMAT)
print(contract.SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT)
print(contract.CANDIDATE_INDEX_FORMAT)
print(contract.SEGMENTED_CANDIDATE_INDEX_FORMAT)
PY
)
if [[ ${#candidate_index_formats[@]} -ne 5 ]]; then
    printf 'candidate index format preflight returned incomplete fields\n' >&2
    exit 1
fi
candidate_index_format=${candidate_index_formats[0]}
partial_candidate_index_format=${candidate_index_formats[1]}
segmented_partial_candidate_index_format=${candidate_index_formats[2]}
complete_candidate_index_format=${candidate_index_formats[3]}
segmented_complete_candidate_index_format=${candidate_index_formats[4]}
IFS=$'\t' read -r candidate_index_authority minimum_plan_jobs \
    legacy_plan_job_modulus gate_stage < <(
        semtalk_prereq_candidate_index_authority \
            "$candidate_index_format" \
            "$partial_candidate_index_format" \
            "$segmented_partial_candidate_index_format" \
            "$complete_candidate_index_format" \
            "$segmented_complete_candidate_index_format"
    )
if [[ "$candidate_index_authority" == partial ]]; then
    case "$partition" in
        nonglobal|gate) ;;
        *)
            printf 'partial candidate index is authorized only for nonglobal/gate partitions\n' >&2
            exit 1
            ;;
    esac
else
    case "$partition" in
        nonglobal)
            printf 'nonglobal partition requires the frozen partial authority\n' >&2
            exit 1
            ;;
    esac
fi

if [[ ! -f "$plan_path" || -L "$plan_path" ]]; then
    printf 'candidate plan was not created as a regular file\n' >&2
    exit 1
fi
mapfile -t candidate_plan <"$plan_path"
if ! semtalk_prereq_plan_job_count_valid \
    "${#candidate_plan[@]}" \
    "$minimum_plan_jobs" \
    "$legacy_plan_job_modulus"; then
    printf 'candidate plan lacks its required complete stage/schedule authority\n' >&2
    exit 1
fi

selected_plan=()
gate_jobs_selected=0
for ((plan_index = 0; plan_index < ${#candidate_plan[@]}; plan_index++)); do
    current_plan_index=$plan_index
    plan_line=${candidate_plan[$plan_index]}
    IFS=$'\t' read -r plan_stage _ <<<"$plan_line"
    include=false
    case "$partition" in
        all)
            include=true
            ;;
        0of2|1of2)
            if ((current_plan_index % partition_modulus == partition_remainder)); then
                include=true
            fi
            ;;
        nonglobal)
            [[ "$plan_stage" != global ]] && include=true
            ;;
        global)
            [[ "$plan_stage" == global ]] && include=true
            ;;
        gate)
            # Four deterministic same-stage candidates prove byte-exact
            # equivalence and measure 1-vs-4 throughput for each authority.
            if [[ "$plan_stage" == "$gate_stage" && \
                  "$gate_jobs_selected" -lt 4 ]]; then
                include=true
                ((gate_jobs_selected += 1))
            fi
            ;;
    esac
    if [[ "$include" == true ]]; then
        selected_plan+=("$plan_line")
    fi
done
expected_jobs=${#selected_plan[@]}
if [[ "$expected_jobs" -le 0 ]]; then
    printf 'partition selected no candidate jobs\n' >&2
    exit 1
fi
expected_waves=$(( \
    (expected_jobs + candidates_per_wave - 1) / candidates_per_wave \
))

if (( BASH_VERSINFO[0] < 5 )); then
    printf 'bash >= 5 is required\n' >&2
    exit 1
fi

launcher_pid=$BASHPID
declare -A active_children=()
declare -A child_expected_ppid=()
declare -A child_starttime=()
declare -A child_cmdline_sha256=()
pending_pid=
pending_expected_ppid=
pending_signal_rc=0
launch_registration_in_progress=false
LAST_CHILD_PID=

sha256_argv() {
    printf '%s\0' "$@" | sha256sum | awk '{print $1}'
}

read_proc_identity() {
    local pid=$1
    local stat_line stat_tail
    local -a stat_fields
    [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1
    stat_line=$(<"/proc/$pid/stat") || return 1
    [[ "$stat_line" == *") "* ]] || return 1
    stat_tail=${stat_line##*) }
    read -r -a stat_fields <<<"$stat_tail"
    (( ${#stat_fields[@]} >= 20 )) || return 1
    PROC_STATE=${stat_fields[0]}
    PROC_PPID=${stat_fields[1]}
    PROC_STARTTIME=${stat_fields[19]}
    PROC_CMDLINE_SHA256=$(sha256sum "/proc/$pid/cmdline" | awk '{print $1}') \
        || return 1
}

capture_child_identity() {
    local pid=$1
    local expected_ppid=$2
    local expected_sha=$3
    local attempt
    for ((attempt = 0; attempt < 500; attempt++)); do
        if ! read_proc_identity "$pid"; then
            printf 'child %s exited before identity capture\n' "$pid" >&2
            return 1
        fi
        if [[ "$PROC_PPID" != "$expected_ppid" || "$PROC_STATE" == Z ]]; then
            printf 'child %s ancestry/state changed during capture\n' "$pid" >&2
            return 1
        fi
        if [[ "$PROC_CMDLINE_SHA256" == "$expected_sha" ]]; then
            child_expected_ppid["$pid"]=$expected_ppid
            child_starttime["$pid"]=$PROC_STARTTIME
            child_cmdline_sha256["$pid"]=$PROC_CMDLINE_SHA256
            active_children["$pid"]=1
            return 0
        fi
        sleep 0.02
    done
    printf 'child %s never reached exact argv\n' "$pid" >&2
    return 1
}

validate_child_identity() {
    local pid=$1
    if ! read_proc_identity "$pid"; then
        return 2
    fi
    if [[ "$PROC_STATE" == Z ]]; then
        return 3
    fi
    if [[ "$PROC_PPID" != "${child_expected_ppid[$pid]}" || \
          "$PROC_STARTTIME" != "${child_starttime[$pid]}" || \
          "$PROC_CMDLINE_SHA256" != "${child_cmdline_sha256[$pid]}" ]]; then
        printf 'refusing TERM for child identity mismatch pid=%s\n' "$pid" >&2
        return 1
    fi
}

terminate_uncaptured_child() {
    local pid=$1
    local expected_ppid=$2
    local previous_start= previous_cmd_sha=
    local attempt
    for ((attempt = 0; attempt < 500; attempt++)); do
        if ! read_proc_identity "$pid"; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        if [[ "$PROC_STATE" == Z ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        if [[ "$PROC_PPID" != "$expected_ppid" ]]; then
            printf 'refusing TERM: uncaptured ancestry changed pid=%s\n' "$pid" >&2
            return 1
        fi
        if [[ -n "$previous_start" && \
              "$PROC_STARTTIME" != "$previous_start" ]]; then
            printf 'refusing TERM: pending PID reused pid=%s\n' "$pid" >&2
            return 1
        fi
        if [[ "$PROC_STARTTIME" == "$previous_start" && \
              "$PROC_CMDLINE_SHA256" == "$previous_cmd_sha" ]]; then
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        previous_start=$PROC_STARTTIME
        previous_cmd_sha=$PROC_CMDLINE_SHA256
        sleep 0.02
    done
    printf 'refusing TERM: pending argv never stabilized pid=%s\n' "$pid" >&2
    return 1
}

cleanup_pending_child() {
    local pid=${pending_pid:-}
    [[ -n "$pid" ]] || return 0
    if [[ -n "${active_children[$pid]+present}" ]]; then
        pending_pid=
        pending_expected_ppid=
        return 0
    fi
    if [[ -z "${pending_expected_ppid:-}" ]]; then
        return 1
    fi
    terminate_uncaptured_child "$pid" "$pending_expected_ppid"
    pending_pid=
    pending_expected_ppid=
}

terminate_children() {
    local pid identity_rc cleanup_rc=0
    local -a snapshot=("${!active_children[@]}")
    local -a term_sent=()
    for pid in "${snapshot[@]}"; do
        if validate_child_identity "$pid"; then
            kill -TERM "$pid"
            term_sent+=("$pid")
            continue
        fi
        identity_rc=$?
        if [[ "$identity_rc" -eq 2 || "$identity_rc" -eq 3 ]]; then
            wait "$pid" 2>/dev/null || true
        else
            cleanup_rc=1
        fi
        unset 'active_children[$pid]'
    done
    for pid in "${term_sent[@]}"; do
        wait "$pid" 2>/dev/null || true
        unset 'active_children[$pid]'
    done
    return "$cleanup_rc"
}

on_signal() {
    local rc=$1
    local cleanup_rc=0
    if [[ "$launch_registration_in_progress" == true && \
          -z "${pending_pid:-}" ]]; then
        pending_signal_rc=$rc
        return
    fi
    trap - EXIT INT TERM
    cleanup_pending_child || cleanup_rc=$?
    terminate_children || cleanup_rc=$?
    if [[ "$cleanup_rc" -ne 0 ]]; then
        printf 'signal cleanup failed rc=%s\n' "$cleanup_rc" >&2
    fi
    exit "$rc"
}

on_exit() {
    local rc=$?
    local cleanup_rc=0
    trap - EXIT INT TERM
    cleanup_pending_child || cleanup_rc=$?
    terminate_children || cleanup_rc=$?
    if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then
        rc=$cleanup_rc
    fi
    exit "$rc"
}

trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

launch_registered() {
    local gpu=$1
    local log_path=$2
    shift 2
    local -a argv=("$@")
    local expected_sha
    expected_sha=$(sha256_argv "${argv[@]}")
    launch_registration_in_progress=true
    pending_expected_ppid=$launcher_pid
    if [[ "$gpu" == cpu ]]; then
        "${argv[@]}" >"$log_path" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu" \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PYTHONHASHSEED=20260731 \
            "${argv[@]}" >"$log_path" 2>&1 &
    fi
    pending_pid=$!
    launch_registration_in_progress=false
    if (( pending_signal_rc != 0 )); then
        on_signal "$pending_signal_rc"
    fi
    capture_child_identity \
        "$pending_pid" "$pending_expected_ppid" "$expected_sha"
    LAST_CHILD_PID=$pending_pid
    pending_pid=
    pending_expected_ppid=
}

wait_registered() {
    local pid=$1
    local rc
    set +e
    wait "$pid"
    rc=$?
    set -e
    unset 'active_children[$pid]'
    return "$rc"
}

started_unix=$(date +%s)
job_number=0
wave_number=0
for ((wave_start = 0; wave_start < expected_jobs; \
      wave_start += candidates_per_wave)); do
    wave_pids=()
    wave_labels=()
    wave_jobs=0
    wave_stop=$((wave_start + candidates_per_wave))
    if ((wave_stop > expected_jobs)); then
        wave_stop=$expected_jobs
    fi
    for ((candidate_offset = wave_start; \
          candidate_offset < wave_stop; candidate_offset++)); do
        plan_line=${selected_plan[$candidate_offset]}
        IFS=$'\t' read -r stage epoch updates checkpoint checkpoint_sha \
            <<<"$plan_line"
        if [[ -z "$stage" || -z "$epoch" || -z "$updates" || \
              -z "$checkpoint" || -z "$checkpoint_sha" ]]; then
            printf 'malformed candidate plan line\n' >&2
            exit 1
        fi
        epoch_dir="$shard_root/shards/$stage/epoch_$(printf '%04d' "$epoch")"
        mkdir -p "$epoch_dir"
        wave_labels+=("${stage}:e${epoch}")
        ((wave_jobs += 1))
        for shard_index in 0 1 2 3 4 5 6 7; do
            output_json="$epoch_dir/shard_$(printf '%02d' "$shard_index").json"
            log_path="$shard_root/logs/${stage}_e$(printf '%04d' "$epoch")_s$(printf '%02d' "$shard_index").log"
            shard_argv=(
                "$python_bin" "$evaluator"
                --stage "$stage"
                --epoch "$epoch"
                --optimizer-updates "$updates"
                --checkpoint "$checkpoint"
                --expected-checkpoint-sha256 "$checkpoint_sha"
                --canonical-manifest "$canonical_manifest"
                --expected-manifest-sha256 "$manifest_sha256"
                --canonical-summary "$canonical_summary"
                --expected-summary-sha256 "$summary_sha256"
                --canonical-lineage "$canonical_lineage"
                --expected-lineage-sha256 "$lineage_sha256"
                --shard-index "$shard_index"
                --shard-count 8
                --device cuda:0
                --batch-size "$batch_size"
                --seed 20260731
                --expected-source-commit "$source_commit"
                --expected-source-tree "$source_tree"
                --output-json "$output_json"
            )
            launch_registered \
                "$shard_index" "$log_path" "${shard_argv[@]}"
            wave_pids+=("$LAST_CHILD_PID")
        done
    done
    wave_rc=0
    for pid in "${wave_pids[@]}"; do
        if ! wait_registered "$pid"; then
            wave_rc=1
        fi
    done
    if [[ "$wave_rc" -ne 0 ]]; then
        printf 'validation wave failed: %s\n' "${wave_labels[*]}" >&2
        exit 1
    fi
    ((job_number += wave_jobs))
    ((wave_number += 1))
    printf 'completed prerequisite validation wave %d/%d jobs=%d/%d: %s\n' \
        "$wave_number" "$expected_waves" "$job_number" "$expected_jobs" \
        "${wave_labels[*]}"
done

if [[ "$job_number" -ne "$expected_jobs" || \
      "$wave_number" -ne "$expected_waves" ]]; then
    printf 'partition coverage mismatch: jobs=%s/%s waves=%s/%s\n' \
        "$job_number" "$expected_jobs" "$wave_number" "$expected_waves" >&2
    exit 1
fi
ended_unix=$(date +%s)
partition_receipt="$shard_root/partition_receipt.json"
"$python_bin" - "$partition_receipt" "$partition" \
    "$candidates_per_wave" "$batch_size" "$job_number" "$wave_number" \
    "$started_unix" "$ended_unix" "$candidate_index" \
    "$candidate_index_sha256" "$manifest_sha256" "$summary_sha256" \
    "$lineage_sha256" "$source_commit" "$source_tree" \
    "$gate_receipt" "$gate_receipt_sha256" \
    "${selected_plan[@]}" <<'PY'
from pathlib import Path
import sys

from scripts.show_base import prerequisite_val_contract as contract

(
    output,
    partition,
    candidates_per_wave,
    batch_size,
    candidate_jobs,
    waves,
    started_unix,
    ended_unix,
    candidate_index,
    candidate_index_sha256,
    manifest_sha256,
    summary_sha256,
    lineage_sha256,
    source_commit,
    source_tree,
    gate_receipt,
    gate_receipt_sha256,
    *plan,
) = sys.argv[1:]
jobs = []
for row in plan:
    fields = row.split("\t")
    if len(fields) != 5:
        raise SystemExit("partition receipt plan is malformed")
    stage, epoch, updates, checkpoint, checkpoint_sha256 = fields
    jobs.append(
        {
            "stage": stage,
            "epoch": int(epoch),
            "optimizer_updates": int(updates),
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
started = int(started_unix)
ended = int(ended_unix)
payload = contract.receipt_payload(
    {
        "format": "semtalk_show_prerequisite_val_partition_v1",
        "status": "complete",
        "partition": partition,
        "test_visible": False,
        "protocol": {
            "candidates_per_wave": int(candidates_per_wave),
            "shards_per_candidate": contract.EXPECTED_SHARDS,
            "batch_size": int(batch_size),
            "candidate_jobs": int(candidate_jobs),
            "waves": int(waves),
        },
        "timing": {
            "started_unix": started,
            "ended_unix": ended,
            "elapsed_seconds": ended - started,
        },
        "inputs": {
            "candidate_index": str(Path(candidate_index).resolve(strict=True)),
            "candidate_index_sha256": candidate_index_sha256,
            "canonical_manifest_sha256": manifest_sha256,
            "canonical_summary_sha256": summary_sha256,
            "canonical_lineage_sha256": lineage_sha256,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "multi_candidate_gate": (
                {
                    "path": str(Path(gate_receipt).resolve(strict=True)),
                    "sha256": gate_receipt_sha256,
                }
                if gate_receipt
                else None
            ),
        },
        "jobs": jobs,
        "coverage": {
            "candidate_jobs": len(jobs),
            "shard_jobs": len(jobs) * contract.EXPECTED_SHARDS,
            "exact_once": True,
        },
    }
)
contract.atomic_json_new(Path(output), payload)
PY
if [[ "$partition" != all ]]; then
    printf 'SHOW prerequisite shard partition complete: partition=%s jobs=%s waves=%s root=%s\n' \
        "$partition" "$job_number" "$wave_number" "$shard_root"
    exit 0
fi

merge_log="$shard_root/logs/merge.log"
merge_argv=(
    "$python_bin" "$merger"
    --candidate-index "$candidate_index"
    --expected-candidate-index-sha256 "$candidate_index_sha256"
    --canonical-manifest "$canonical_manifest"
    --expected-manifest-sha256 "$manifest_sha256"
    --canonical-summary "$canonical_summary"
    --expected-summary-sha256 "$summary_sha256"
    --canonical-lineage "$canonical_lineage"
    --expected-lineage-sha256 "$lineage_sha256"
    --shard-root "$shard_root"
    --output-root "$measurement_root"
    --expected-source-commit "$source_commit"
    --expected-source-tree "$source_tree"
)
launch_registered cpu "$merge_log" "${merge_argv[@]}"
if ! wait_registered "$LAST_CHILD_PID"; then
    printf 'strict prerequisite validation merge failed\n' >&2
    exit 1
fi

measurement_index="$measurement_root/measurement_index.json"
if [[ ! -f "$measurement_index" || -L "$measurement_index" ]]; then
    printf 'merge did not create the measurement index\n' >&2
    exit 1
fi
measurement_index_sha256=$(
    sha256sum "$measurement_index" | awk '{print $1}'
)
select_log="$shard_root/logs/select.log"
select_argv=(
    "$python_bin" "$selector"
    --candidate-index "$candidate_index"
    --expected-candidate-index-sha256 "$candidate_index_sha256"
    --measurement-index "$measurement_index"
    --expected-measurement-index-sha256 "$measurement_index_sha256"
    --output-json "$selection_json"
    --expected-source-commit "$source_commit"
    --expected-source-tree "$source_tree"
)
launch_registered cpu "$select_log" "${select_argv[@]}"
if ! wait_registered "$LAST_CHILD_PID"; then
    printf 'strict prerequisite candidate selection failed\n' >&2
    exit 1
fi

selection_sha256=$(sha256sum "$selection_json" | awk '{print $1}')
printf 'SHOW prerequisite validation selected successfully: %s sha256=%s\n' \
    "$selection_json" "$selection_sha256"
