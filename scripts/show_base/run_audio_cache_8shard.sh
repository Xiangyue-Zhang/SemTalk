#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Run only as the direct workload of /tmp/globaldiff_guarded_runner.py with
# GPUs 0..7 reserved.  Every GPU builds one immutable modulo shard.

if [[ $# -ne 17 ]]; then
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON CANONICAL_MANIFEST CANONICAL_SUMMARY CANONICAL_LINEAGE HUBERT_MODEL HUBERT_TREE_SHA256 OUTPUT_ROOT SPLIT EXPECTED_CLIPS SOURCE_COMMIT SOURCE_TREE CANONICAL_SOURCE_COMMIT CANONICAL_SOURCE_TREE CANONICAL_MANIFEST_SHA256 CANONICAL_SUMMARY_SHA256 CANONICAL_LINEAGE_SHA256"
    exit 2
fi

repo_root=$1
python_bin=$2
canonical_manifest=$3
canonical_summary=$4
canonical_lineage=$5
hubert_model=$6
hubert_tree_sha256=$7
output_root=$8
split=$9
expected_clips=${10}
source_commit=${11}
source_tree=${12}
canonical_source_commit=${13}
canonical_source_tree=${14}
canonical_manifest_sha256=${15}
canonical_summary_sha256=${16}
canonical_lineage_sha256=${17}
builder="$repo_root/scripts/show_base/build_base_features.py"

if [[ "$split" != train && "$split" != test ]]; then
    printf 'split must be train or test, got: %s\n' "$split" >&2
    exit 2
fi
for frozen_sha in "$canonical_manifest_sha256" "$canonical_summary_sha256" \
    "$canonical_lineage_sha256"; do
    if [[ ! "$frozen_sha" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'invalid frozen canonical SHA-256: %s\n' "$frozen_sha" >&2
        exit 2
    fi
done

for required in "$builder" "$python_bin" "$canonical_manifest" \
    "$canonical_summary" "$canonical_lineage" "$hubert_model"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ -e "$output_root" ]]; then
    printf 'refusing to reuse audio cache root: %s\n' "$output_root" >&2
    exit 1
fi
mkdir -p "$output_root"

if (( BASH_VERSINFO[0] < 5 || \
      (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    printf 'bash >= 5.1 is required for wait -n -p\n' >&2
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
        if [[ "$PROC_PPID" != "$expected_ppid" ]]; then
            printf 'child %s unexpected PPID during identity capture: expected=%s actual=%s\n' \
                "$pid" "$expected_ppid" "$PROC_PPID" >&2
            return 1
        fi
        if [[ "$PROC_STATE" == Z ]]; then
            printf 'child %s became a zombie before exec identity capture\n' "$pid" >&2
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

    printf 'child %s did not exec to the exact expected builder argv\n' "$pid" >&2
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
        printf '%s\n' \
            "refusing TERM for identity mismatch:" \
            "pid=$pid" \
            "expected_ppid=${child_expected_ppid[$pid]} actual_ppid=$PROC_PPID" \
            "expected_starttime=${child_starttime[$pid]} actual_starttime=$PROC_STARTTIME" \
            "expected_cmdline_sha256=${child_cmdline_sha256[$pid]} actual_cmdline_sha256=$PROC_CMDLINE_SHA256" \
            >&2
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
            printf 'refusing TERM: uncaptured child ancestry changed pid=%s\n' \
                "$pid" >&2
            return 1
        fi
        if [[ -n "$previous_start" && \
              "$PROC_STARTTIME" != "$previous_start" ]]; then
            printf 'refusing TERM: pending PID starttime changed pid=%s\n' \
                "$pid" >&2
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
    printf 'refusing TERM: pending child argv never stabilized pid=%s\n' \
        "$pid" >&2
    return 1
}

cleanup_pending_child() {
    local pid=${pending_pid:-}
    local expected_ppid=${pending_expected_ppid:-}
    [[ -n "$pid" ]] || return 0
    if [[ -n "${active_children[$pid]+present}" ]]; then
        pending_pid=
        pending_expected_ppid=
        return 0
    fi
    if [[ -z "$expected_ppid" ]]; then
        printf 'pending child is missing its exact expected PPID: pid=%s\n' \
            "$pid" >&2
        return 1
    fi
    if terminate_uncaptured_child "$pid" "$expected_ppid"; then
        pending_pid=
        pending_expected_ppid=
        return 0
    fi
    printf 'failed to clean exact pending child pid=%s\n' "$pid" >&2
    return 1
}

terminate_children() {
    local pid identity_rc cleanup_rc=0
    local -a active_snapshot=("${!active_children[@]}")
    local -a term_sent=()

    for pid in "${active_snapshot[@]}"; do
        if validate_child_identity "$pid"; then
            if kill -TERM "$pid" 2>/dev/null; then
                term_sent+=("$pid")
            fi
            continue
        fi
        identity_rc=$?
        if [[ "$identity_rc" -eq 2 || "$identity_rc" -eq 3 ]]; then
            wait "$pid" 2>/dev/null || true
            unset 'active_children[$pid]'
        else
            cleanup_rc=1
            unset 'active_children[$pid]'
        fi
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

for shard_id in 0 1 2 3 4 5 6 7; do
    shard_root="$output_root/shard_${shard_id}"
    mkdir -p "$shard_root"
    shard_argv=(
        "$python_bin" "$builder" audio
        --canonical-manifest "$canonical_manifest"
        --canonical-summary "$canonical_summary"
        --canonical-lineage "$canonical_lineage"
        --split "$split"
        --hubert-model "$hubert_model"
        --expected-hubert-tree-sha256 "$hubert_tree_sha256"
        --output-dir "$shard_root/features"
        --output-manifest "$shard_root/manifest.jsonl"
        --summary-json "$shard_root/summary.json"
        --lineage-json "$shard_root/lineage.json"
        --device cuda:0
        --shard-id "$shard_id"
        --num-shards 8
        --expected-total-clips "$expected_clips"
        --expected-source-commit "$source_commit"
        --expected-source-tree "$source_tree"
        --expected-canonical-source-commit "$canonical_source_commit"
        --expected-canonical-source-tree "$canonical_source_tree"
        --expected-canonical-manifest-sha256 "$canonical_manifest_sha256"
        --expected-canonical-summary-sha256 "$canonical_summary_sha256"
        --expected-canonical-lineage-sha256 "$canonical_lineage_sha256"
        --max-frame-mismatch 1
    )
    expected_cmdline_sha256=$(sha256_argv "${shard_argv[@]}")
    launch_registration_in_progress=true
    pending_expected_ppid=$launcher_pid
    (
        export CUDA_VISIBLE_DEVICES="$shard_id"
        cd "$repo_root"
        exec "${shard_argv[@]}"
    ) >"$shard_root/run.log" 2>&1 &
    child_pid=$!
    pending_pid=$child_pid
    launch_registration_in_progress=false
    if ((pending_signal_rc != 0)); then
        deferred_signal_rc=$pending_signal_rc
        pending_signal_rc=0
        on_signal "$deferred_signal_rc"
    fi
    if ! capture_child_identity \
        "$child_pid" "$launcher_pid" "$expected_cmdline_sha256"; then
        pending_cleanup_rc=0
        cleanup_pending_child || pending_cleanup_rc=$?
        terminate_children || true
        if ((pending_cleanup_rc != 0)); then
            printf 'pending child cleanup failed rc=%s\n' \
                "$pending_cleanup_rc" >&2
        fi
        exit 1
    fi
    pending_pid=
    pending_expected_ppid=
done

while (( ${#active_children[@]} > 0 )); do
    active_pids=("${!active_children[@]}")
    completed_pid=
    if wait -n -p completed_pid "${active_pids[@]}"; then
        child_rc=0
    else
        child_rc=$?
    fi
    if [[ -z "$completed_pid" || -z "${active_children[$completed_pid]+present}" ]]; then
        printf 'wait -n did not identify an active child\n' >&2
        terminate_children
        exit 1
    fi
    unset 'active_children[$completed_pid]'
    if [[ "$child_rc" -ne 0 ]]; then
        printf 'audio cache shard child failed: pid=%s rc=%s\n' \
            "$completed_pid" "$child_rc" >&2
        terminate_children || true
        exit "$child_rc"
    fi
done

trap - EXIT INT TERM
exit 0
