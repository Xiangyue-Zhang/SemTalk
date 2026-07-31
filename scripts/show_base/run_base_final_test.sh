#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# One-shot formal launcher for the authority-consuming SHOW test adapter.
# The direct parent must be /tmp/globaldiff_guarded_runner.py reserving exactly
# GPU0..7.  Each child sees exactly one physical GPU as cuda:0; its Linux PID,
# direct PPID, starttime and full argv hash are recorded before it is trusted.

launcher_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=guarded_runner_contract.sh
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

python_bin=
fresh_test_authority=
authority_sha=
authority_bytes=
authority_payload_sha=
paspa_root=
diffsheg_root=
talkshow_root=
source_audio_root=
smplx_path=
seed=20260801
diffsheg_batch_size=64

while (($#)); do
    case "$1" in
        --python)
            python_bin=$2; shift 2 ;;
        --fresh-test-authority)
            fresh_test_authority=$2; shift 2 ;;
        --expected-test-authority-sha256)
            authority_sha=$2; shift 2 ;;
        --expected-test-authority-bytes)
            authority_bytes=$2; shift 2 ;;
        --expected-test-authority-receipt-payload-sha256)
            authority_payload_sha=$2; shift 2 ;;
        --paspa-root)
            paspa_root=$2; shift 2 ;;
        --diffsheg-root)
            diffsheg_root=$2; shift 2 ;;
        --talkshow-root)
            talkshow_root=$2; shift 2 ;;
        --source-audio-root)
            source_audio_root=$2; shift 2 ;;
        --smplx-path)
            smplx_path=$2; shift 2 ;;
        --seed)
            seed=$2; shift 2 ;;
        --diffsheg-batch-size)
            diffsheg_batch_size=$2; shift 2 ;;
        --)
            shift
            (($# == 0)) || {
                printf 'unexpected positional arguments after --\n' >&2
                exit 2
            }
            ;;
        *)
            printf 'unknown or forbidden argument: %s\n' "$1" >&2
            exit 2 ;;
    esac
done

for value_name in \
    python_bin fresh_test_authority authority_sha authority_bytes \
    authority_payload_sha paspa_root \
    diffsheg_root talkshow_root source_audio_root smplx_path; do
    if [[ -z ${!value_name} ]]; then
        printf 'missing required option value: %s\n' "$value_name" >&2
        exit 2
    fi
done
for digest in "$authority_sha" "$authority_payload_sha"; do
    [[ $digest =~ ^[0-9a-f]{64}$ ]] || {
        printf 'invalid lowercase SHA-256: %s\n' "$digest" >&2
        exit 2
    }
done
for integer in "$authority_bytes"; do
    [[ $integer =~ ^[1-9][0-9]*$ ]] || {
        printf 'invalid positive byte count: %s\n' "$integer" >&2
        exit 2
    }
done
[[ $seed =~ ^[0-9]+$ && $diffsheg_batch_size =~ ^[1-9][0-9]*$ ]] || {
    printf 'seed/diffsheg-batch-size contract mismatch\n' >&2
    exit 2
}

adapter=$launcher_dir/run_base_final_test.py
evaluator=$launcher_dir/evaluate_diffsheg_final_test.py
audio_view_builder=$launcher_dir/prepare_diffsheg_audio_view.py
for required in \
    "$python_bin" "$fresh_test_authority" \
    "$paspa_root" "$diffsheg_root" "$talkshow_root" \
    "$source_audio_root" "$smplx_path" "$adapter" "$evaluator" \
    "$audio_view_builder"; do
    [[ -e $required ]] || {
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    }
done

fresh_test_authority=$(realpath -e -- "$fresh_test_authority")
paspa_root=$(realpath -e -- "$paspa_root")
diffsheg_root=$(realpath -e -- "$diffsheg_root")
talkshow_root=$(realpath -e -- "$talkshow_root")
source_audio_root=$(realpath -e -- "$source_audio_root")
smplx_path=$(realpath -e -- "$smplx_path")
python_bin=$(realpath -e -- "$python_bin")

authority_args=(
    --fresh-test-authority "$fresh_test_authority"
    --expected-test-authority-sha256 "$authority_sha"
    --expected-test-authority-bytes "$authority_bytes"
    --expected-test-authority-receipt-payload-sha256 "$authority_payload_sha"
    --seed "$seed"
)

preflight_temp=$(mktemp -d /tmp/semtalk-final-test-preflight.XXXXXXXX)
cleanup_preflight() {
    rm -rf -- "$preflight_temp"
}
preflight_signal_exit() {
    local code=$1
    trap - EXIT INT TERM HUP
    cleanup_preflight
    exit "$code"
}
trap cleanup_preflight EXIT
trap 'preflight_signal_exit 130' INT
trap 'preflight_signal_exit 143' TERM
trap 'preflight_signal_exit 129' HUP
prepared_temp=$preflight_temp/prepared-authority.json
context_fields=()
mapfile -d '' -t context_fields < <(
    "$python_bin" "$adapter" context \
        "${authority_args[@]}" \
        --prepared-authority-output "$prepared_temp" \
        --nul-context
)
if ((${#context_fields[@]} != 10)); then
    printf 'authority context did not return ten exact fields\n' >&2
    exit 1
fi
output_root=${context_fields[0]}
shards_root=${context_fields[1]}
canonical_manifest=${context_fields[2]}
canonical_manifest_sha=${context_fields[3]}
source_commit=${context_fields[4]}
source_tree=${context_fields[5]}
prepared_path=${context_fields[6]}
prepared_sha=${context_fields[7]}
prepared_bytes=${context_fields[8]}
prepared_payload_sha=${context_fields[9]}

[[ $source_commit =~ ^[0-9a-f]{40}$ && $source_tree =~ ^[0-9a-f]{40}$ ]] || {
    printf 'authority source commit/tree is invalid\n' >&2
    exit 1
}
[[ ! -e $output_root && ! -L $output_root && ! -e $shards_root && ! -L $shards_root ]] || {
    printf 'one-shot output namespace is already present\n' >&2
    exit 1
}

log_root="$(dirname -- "$output_root")/.semtalk-final-test-${authority_sha:0:16}"
if [[ -e $log_root || -L $log_root ]]; then
    printf 'one-shot log namespace is already present: %s\n' "$log_root" >&2
    exit 1
fi
mkdir -m 700 -- "$log_root"
prepared_authority=$log_root/prepared-authority.json
cp -- "$prepared_path" "$prepared_authority"
copied_prepared_sha=$(sha256sum "$prepared_authority")
copied_prepared_sha=${copied_prepared_sha%% *}
copied_prepared_bytes=$(stat -c %s "$prepared_authority")
if [[ $copied_prepared_sha != "$prepared_sha" || \
      $copied_prepared_bytes != "$prepared_bytes" ]]; then
    printf 'prepared authority copy changed\n' >&2
    exit 1
fi
rm -rf -- "$preflight_temp"
trap - EXIT INT TERM HUP

workload_authority_args=(
    "${authority_args[@]}"
    --prepared-authority "$prepared_authority"
    --expected-prepared-authority-sha256 "$prepared_sha"
    --expected-prepared-authority-bytes "$prepared_bytes"
    --expected-prepared-authority-receipt-payload-sha256 \
        "$prepared_payload_sha"
)

proc_snapshot() {
    local pid=$1
    local stat_line stat_tail argv_sha
    local -a fields
    [[ $pid =~ ^[1-9][0-9]*$ && -r /proc/$pid/stat && -r /proc/$pid/cmdline ]] || return 1
    stat_line=$(<"/proc/$pid/stat") || return 1
    [[ $stat_line == *") "* ]] || return 1
    stat_tail=${stat_line##*) }
    read -r -a fields <<<"$stat_tail"
    ((${#fields[@]} >= 20)) || return 1
    [[ ${fields[0]} != Z && ${fields[1]} =~ ^[1-9][0-9]*$ && ${fields[19]} =~ ^[0-9]+$ ]] || return 1
    argv_sha=$(sha256sum "/proc/$pid/cmdline") || return 1
    argv_sha=${argv_sha%% *}
    printf '%s %s %s\n' "${fields[1]}" "${fields[19]}" "$argv_sha"
}

proc_exact_python_entry() {
    local pid=$1 entrypoint=$2 subcommand=${3-}
    local -a argv
    [[ -r /proc/$pid/cmdline ]] || return 1
    mapfile -d '' -t argv <"/proc/$pid/cmdline" || return 1
    ((${#argv[@]} >= 2)) || return 1
    [[ ${argv[0]} == "$python_bin" && ${argv[1]} == "$entrypoint" ]] || return 1
    if [[ -n $subcommand ]]; then
        ((${#argv[@]} >= 3)) || return 1
        [[ ${argv[2]} == "$subcommand" ]] || return 1
    fi
}

pids=()
starttimes=()
argv_hashes=()
pending_pid=
pending_entry=
pending_subcommand=

verified_live_child() {
    local index=$1 snapshot ppid starttime argv_sha
    snapshot=$(proc_snapshot "${pids[$index]}") || return 1
    read -r ppid starttime argv_sha <<<"$snapshot"
    [[ $ppid == $$ && $starttime == "${starttimes[$index]}" && $argv_sha == "${argv_hashes[$index]}" ]]
}

terminate_pending_child() {
    local first second ppid
    [[ -n $pending_pid ]] || return 0
    for _attempt in {1..200}; do
        first=$(proc_snapshot "$pending_pid") || first=
        if [[ -n $first ]]; then
            read -r ppid _ _ <<<"$first"
            second=$(proc_snapshot "$pending_pid") || second=
        fi
        if [[ -n $first && $first == "$second" && $ppid == $$ ]]; then
            # The exact current identity is now known even if exec has not yet
            # reached the expected Python argv.  PID reuse is excluded by the
            # unchanged PPID/starttime/full-cmdline hash pair.
            kill -TERM -- "$pending_pid" 2>/dev/null || true
            wait "$pending_pid" 2>/dev/null || true
            pending_pid=
            pending_entry=
            pending_subcommand=
            return 0
        fi
        if [[ ! -e /proc/$pending_pid ]]; then
            wait "$pending_pid" 2>/dev/null || true
            pending_pid=
            pending_entry=
            pending_subcommand=
            return 0
        fi
        sleep 0.01
    done
    if [[ -e /proc/$pending_pid ]]; then
        printf 'guarded runner must clean unattestable pending child %s\n' \
            "$pending_pid" >&2
        return 0
    fi
    wait "$pending_pid" 2>/dev/null || true
    pending_pid=
    pending_entry=
    pending_subcommand=
}

cleanup_children() {
    local index
    terminate_pending_child
    for index in "${!pids[@]}"; do
        if verified_live_child "$index"; then
            kill -TERM -- "${pids[$index]}" 2>/dev/null || true
        fi
    done
    for index in "${!pids[@]}"; do
        wait "${pids[$index]}" 2>/dev/null || true
    done
}

signal_exit() {
    local code=$1
    trap - EXIT INT TERM HUP
    cleanup_children
    exit "$code"
}

trap cleanup_children EXIT
trap 'signal_exit 130' INT
trap 'signal_exit 143' TERM
trap 'signal_exit 129' HUP

for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu "$python_bin" "$adapter" shard \
        "${workload_authority_args[@]}" \
        --shard-id "$gpu" \
        --device cuda:0 \
        >"$log_root/shard-$gpu.log" 2>&1 &
    pid=$!
    pending_pid=$pid
    pending_entry=$adapter
    pending_subcommand=shard
    snapshot=
    for _attempt in {1..200}; do
        if proc_exact_python_entry "$pid" "$adapter" shard && \
           snapshot=$(proc_snapshot "$pid") && \
           proc_exact_python_entry "$pid" "$adapter" shard; then
            read -r observed_ppid observed_start observed_argv <<<"$snapshot"
            if [[ $observed_ppid == $$ ]]; then
                break
            fi
        fi
        snapshot=
        sleep 0.01
    done
    if [[ -z $snapshot ]]; then
        printf 'cannot attest shard child %s\n' "$gpu" >&2
        exit 1
    fi
    pids+=("$pid")
    starttimes+=("$observed_start")
    argv_hashes+=("$observed_argv")
    pending_pid=
    pending_entry=
    pending_subcommand=
    printf '%s\t%s\t%s\t%s\n' \
        "$gpu" "$pid" "$observed_start" "$observed_argv" \
        >>"$log_root/children.tsv"
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        failed=1
        break
    fi
    pids[$index]=0
done
if ((failed)); then
    printf 'at least one exact shard child failed\n' >&2
    exit 1
fi
pids=()
starttimes=()
argv_hashes=()
trap - EXIT INT TERM HUP

"$python_bin" "$adapter" finalize "${workload_authority_args[@]}" \
    >"$log_root/finalize.log" 2>&1

diffsheg_preflight=$log_root/diffsheg-preflight.json
diffsheg_output=$output_root/diffsheg-final-metrics
diffsheg_audio_view=$log_root/diffsheg-audio-view
audio_view_result=$(
    "$python_bin" "$audio_view_builder" \
        "${workload_authority_args[@]}" \
        --inference-final-root "$output_root" \
        --source-audio-root "$source_audio_root" \
        --output-root "$diffsheg_audio_view"
)
read -r audio_view_root audio_view_receipt_sha audio_view_source_set_sha \
    audio_view_sealed_set_sha < <(
    "$python_bin" -c \
        'import json,sys; v=json.load(sys.stdin); print(v["output_root"],v["receipt_sha256"],v["source_ordered_set_sha256"],v["view_ordered_set_sha256"])' \
        <<<"$audio_view_result"
)
[[ $audio_view_root == "$diffsheg_audio_view" && \
   $audio_view_receipt_sha =~ ^[0-9a-f]{64}$ && \
   $audio_view_source_set_sha =~ ^[0-9a-f]{64}$ && \
   $audio_view_sealed_set_sha =~ ^[0-9a-f]{64}$ ]] || {
    printf 'sealed DiffSHEG audio view did not return exact pins\n' >&2
    exit 1
}
diffsheg_asset_args=(
    --inference-final-root "$output_root"
    --paspa-root "$paspa_root"
    --diffsheg-root "$diffsheg_root"
    --talkshow-root "$talkshow_root"
    --source-audio-root "$diffsheg_audio_view"
    --smplx-path "$smplx_path"
    --batch-size "$diffsheg_batch_size"
)

# CPU-only preflight consumes no test metric allowance.  It is published once,
# externally pinned below, and replayed by the sole formal DiffSHEG process.
preflight_result=$(
    "$python_bin" "$evaluator" \
        "${workload_authority_args[@]}" \
        "${diffsheg_asset_args[@]}" \
        --preflight-only \
        --output-report "$diffsheg_preflight"
)
read -r diffsheg_preflight_sha diffsheg_audio_set_sha < <(
    "$python_bin" -c \
        'import json,sys; v=json.load(sys.stdin); print(v["sha256"],v["audio_set_sha256"])' \
        <<<"$preflight_result"
)
[[ $diffsheg_preflight_sha =~ ^[0-9a-f]{64}$ && \
   $diffsheg_audio_set_sha =~ ^[0-9a-f]{64}$ ]] || {
    printf 'DiffSHEG preflight did not return exact external pins\n' >&2
    exit 1
}
[[ ! -e $diffsheg_output && ! -L $diffsheg_output ]] || {
    printf 'DiffSHEG output already exists; refusing a second test evaluation\n' >&2
    exit 1
}

# This is the sole formal seven-metric test process.  No compatibility
# evaluator is reachable from this launcher or can consume a claim.
pids=()
starttimes=()
argv_hashes=()
trap cleanup_children EXIT
trap 'signal_exit 130' INT
trap 'signal_exit 143' TERM
trap 'signal_exit 129' HUP
CUDA_VISIBLE_DEVICES=0 "$python_bin" "$evaluator" \
    "${workload_authority_args[@]}" \
    "${diffsheg_asset_args[@]}" \
    --preflight-json "$diffsheg_preflight" \
    --expected-preflight-sha256 "$diffsheg_preflight_sha" \
    --expected-audio-set-sha256 "$diffsheg_audio_set_sha" \
    --output-root "$diffsheg_output" \
    --device cuda:0 \
    >"$log_root/diffsheg-metrics.log" 2>&1 &
metric_pid=$!
pending_pid=$metric_pid
pending_entry=$evaluator
pending_subcommand=
metric_snapshot=
for _attempt in {1..200}; do
    if proc_exact_python_entry "$metric_pid" "$evaluator" && \
       metric_snapshot=$(proc_snapshot "$metric_pid") && \
       proc_exact_python_entry "$metric_pid" "$evaluator"; then
        read -r metric_ppid metric_start metric_argv <<<"$metric_snapshot"
        if [[ $metric_ppid == $$ ]]; then
            break
        fi
    fi
    metric_snapshot=
    sleep 0.01
done
if [[ -z $metric_snapshot ]]; then
    printf 'cannot attest the sole DiffSHEG metric child\n' >&2
    exit 1
fi
pids+=("$metric_pid")
starttimes+=("$metric_start")
argv_hashes+=("$metric_argv")
pending_pid=
pending_entry=
pending_subcommand=
printf 'metric\t%s\t%s\t%s\n' \
    "$metric_pid" "$metric_start" "$metric_argv" \
    >>"$log_root/children.tsv"
if ! wait "$metric_pid"; then
    printf 'the sole DiffSHEG test evaluation failed; no retry is allowed\n' >&2
    exit 1
fi
pids=()
starttimes=()
argv_hashes=()
trap - EXIT INT TERM HUP

[[ -f $diffsheg_output/final_metrics.json && \
   -s $diffsheg_output/final_metrics.json ]] || {
    printf 'formal result is not the unique DiffSHEG seven-metric closure\n' >&2
    exit 1
}
printf 'SemTalk SHOW DiffSHEG final test complete: %s\n' \
    "$diffsheg_output/final_metrics.json"
