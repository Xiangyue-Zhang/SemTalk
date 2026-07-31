#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Formal worker: direct child of one all-GPU globaldiff guarded runner.  Each
# of the two fixed hosts owns eleven epochs, and every epoch is inferred by
# eight direct, argv-registered Python children before one registered FGD
# process creates its immutable report.

usage() {
    printf '%s\n' \
        "Usage: $0 --repo-root PATH --python PATH --partition-id {0,1} --partition-count 2 --run-root NEW_PATH --source-commit OID --source-tree OID --base-candidate-manifest PATH --expected-base-candidate-manifest-sha256 SHA --base-status-json PATH --expected-base-formal-status-sha256 SHA --base-frozen-inputs-json PATH --expected-base-frozen-inputs-sha256 SHA --val-inputs-json PATH --expected-val-inputs-sha256 SHA --pipeline-json PATH --expected-pipeline-sha256 SHA --paspa-root PATH --diffsheg-root PATH --seed INT --diffsheg-batch-size INT"
}

if ((BASH_VERSINFO[0] < 5)); then
    printf 'bash >= 5 is required\n' >&2
    exit 1
fi

declare -A seen_options=()
repo_root=
python_bin=
partition_id=
partition_count=
run_root=
source_commit=
source_tree=
base_candidate_manifest=
expected_base_candidate_manifest_sha256=
base_status_json=
expected_base_formal_status_sha256=
base_frozen_inputs_json=
expected_base_frozen_inputs_sha256=
val_inputs_json=
expected_val_inputs_sha256=
pipeline_json=
expected_pipeline_sha256=
paspa_root=
diffsheg_root=
seed=
diffsheg_batch_size=

set_option() {
    local option=$1 variable=$2 value=$3
    if [[ -n ${seen_options[$option]+present} ]]; then
        printf 'duplicate option: %s\n' "$option" >&2
        exit 2
    fi
    seen_options[$option]=1
    printf -v "$variable" '%s' "$value"
}

while (($#)); do
    if [[ $1 == --help ]]; then
        usage
        exit 0
    fi
    if (($# < 2)); then
        usage >&2
        exit 2
    fi
    case $1 in
        --repo-root) set_option "$1" repo_root "$2" ;;
        --python) set_option "$1" python_bin "$2" ;;
        --partition-id) set_option "$1" partition_id "$2" ;;
        --partition-count) set_option "$1" partition_count "$2" ;;
        --run-root) set_option "$1" run_root "$2" ;;
        --source-commit) set_option "$1" source_commit "$2" ;;
        --source-tree) set_option "$1" source_tree "$2" ;;
        --base-candidate-manifest)
            set_option "$1" base_candidate_manifest "$2" ;;
        --expected-base-candidate-manifest-sha256)
            set_option "$1" expected_base_candidate_manifest_sha256 "$2" ;;
        --base-status-json) set_option "$1" base_status_json "$2" ;;
        --expected-base-formal-status-sha256)
            set_option "$1" expected_base_formal_status_sha256 "$2" ;;
        --base-frozen-inputs-json)
            set_option "$1" base_frozen_inputs_json "$2" ;;
        --expected-base-frozen-inputs-sha256)
            set_option "$1" expected_base_frozen_inputs_sha256 "$2" ;;
        --val-inputs-json) set_option "$1" val_inputs_json "$2" ;;
        --expected-val-inputs-sha256)
            set_option "$1" expected_val_inputs_sha256 "$2" ;;
        --pipeline-json) set_option "$1" pipeline_json "$2" ;;
        --expected-pipeline-sha256)
            set_option "$1" expected_pipeline_sha256 "$2" ;;
        --paspa-root) set_option "$1" paspa_root "$2" ;;
        --diffsheg-root) set_option "$1" diffsheg_root "$2" ;;
        --seed) set_option "$1" seed "$2" ;;
        --diffsheg-batch-size)
            set_option "$1" diffsheg_batch_size "$2" ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift 2
done

for required in \
    repo_root python_bin partition_id partition_count run_root \
    source_commit source_tree base_candidate_manifest \
    expected_base_candidate_manifest_sha256 base_status_json \
    expected_base_formal_status_sha256 base_frozen_inputs_json \
    expected_base_frozen_inputs_sha256 val_inputs_json \
    expected_val_inputs_sha256 pipeline_json expected_pipeline_sha256 \
    paspa_root diffsheg_root seed diffsheg_batch_size; do
    if [[ -z ${!required} ]]; then
        printf 'missing required option value: %s\n' "$required" >&2
        exit 2
    fi
done

if [[ $partition_count != 2 || \
      ($partition_id != 0 && $partition_id != 1) ]]; then
    printf 'formal partition must be exactly 0/2 or 1/2\n' >&2
    exit 2
fi
if [[ ! $seed =~ ^[0-9]+$ || ! $diffsheg_batch_size =~ ^[1-9][0-9]*$ ]]; then
    printf 'seed and DiffSHEG batch size must be positive integers\n' >&2
    exit 2
fi
for digest in \
    "$expected_base_candidate_manifest_sha256" \
    "$expected_base_formal_status_sha256" \
    "$expected_base_frozen_inputs_sha256" \
    "$expected_val_inputs_sha256" "$expected_pipeline_sha256"; do
    [[ $digest =~ ^[0-9a-f]{64}$ ]] || {
        printf 'invalid explicit SHA-256 root\n' >&2
        exit 2
    }
done
for oid in "$source_commit" "$source_tree"; do
    [[ $oid =~ ^[0-9a-f]{40}$ ]] || {
        printf 'invalid source Git OID\n' >&2
        exit 2
    }
done
for path_value in \
    "$repo_root" "$python_bin" "$run_root" "$base_candidate_manifest" \
    "$base_status_json" "$base_frozen_inputs_json" "$val_inputs_json" \
    "$pipeline_json" "$paspa_root" "$diffsheg_root"; do
    [[ $path_value == /* ]] || {
        printf 'every formal path must be absolute\n' >&2
        exit 2
    }
done

raw_repo_root=$repo_root
repo_root=$(realpath -e -- "$repo_root")
if [[ $repo_root != "$raw_repo_root" || ! -d $repo_root || -L $raw_repo_root ]]; then
    printf 'repository root must be canonical and non-symlinked\n' >&2
    exit 1
fi
if [[ ! -x $python_bin || ! -f $python_bin ]]; then
    printf 'Python interpreter is unavailable\n' >&2
    exit 1
fi
run_parent=$(realpath -e -- "$(dirname -- "$run_root")")
if [[ $run_root != "$run_parent/$(basename -- "$run_root")" || \
      -e $run_root || -L $run_root ]]; then
    printf 'formal run root must be one new canonical path\n' >&2
    exit 1
fi
case "$run_root/" in
    "$repo_root/"*)
        printf 'formal output must remain outside the source checkout\n' >&2
        exit 1
        ;;
esac

launcher_name=run_base_diffsheg_val_8shard.sh
if [[ -L ${BASH_SOURCE[0]} ]]; then
    printf 'formal launcher must not be a symlink\n' >&2
    exit 1
fi
launcher_path=$(realpath -e -- "${BASH_SOURCE[0]}")
launcher_dir=$repo_root/scripts/show_base
if [[ $launcher_path != "$launcher_dir/$launcher_name" ]]; then
    printf 'launcher is not the tracked entrypoint in the pinned source\n' >&2
    exit 1
fi
guard_contract=$launcher_dir/guarded_runner_contract.sh
python_runtime_contract=$launcher_dir/formal_python_runtime_contract.sh
partition_contract=$launcher_dir/base_diffsheg_val_partition_contract.py
inference=$launcher_dir/run_base_val_inference.py
evaluator=$launcher_dir/evaluate_diffsheg_val_fgd.py
measurement_producer=$launcher_dir/produce_base_val_measurement.py
long_selector=$launcher_dir/select_base_official_adapt_long.py
for required_path in \
    "$guard_contract" "$python_runtime_contract" "$partition_contract" \
    "$inference" "$evaluator" \
    "$measurement_producer" "$long_selector"; do
    if [[ ! -f $required_path || -L $required_path ]]; then
        printf 'required tracked source is unavailable: %s\n' "$required_path" >&2
        exit 1
    fi
done

. "$guard_contract"
semtalk_require_exact_guarded_runner_all_gpus

formal_host=$(hostname)
case $partition_id in
    0)
        expected_formal_host=
        expected_formal_host+=iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0
        ;;
    1)
        expected_formal_host=
        expected_formal_host+=iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0
        ;;
esac
if [[ $formal_host != "$expected_formal_host" ]]; then
    printf 'partition %s requires host %s, observed %s\n' \
        "$partition_id" "$expected_formal_host" "$formal_host" >&2
    exit 1
fi

if [[ $(git -C "$repo_root" remote get-url origin) != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" rev-parse HEAD) != "$source_commit" || \
      $(git -C "$repo_root" rev-parse 'HEAD^{tree}') != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'formal source must be the exact clean detached zero-branch checkout\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/run_base_diffsheg_val_8shard.sh \
    scripts/show_base/guarded_runner_contract.sh \
    scripts/show_base/formal_python_runtime_contract.sh \
    scripts/show_base/base_diffsheg_val_partition_contract.py \
    scripts/show_base/run_base_val_inference.py \
    scripts/show_base/evaluate_diffsheg_val_fgd.py \
    scripts/show_base/produce_base_val_measurement.py \
    scripts/show_base/select_base_official_adapt_long.py \
    scripts/show_base/base_long_val_contract.py; do
    if [[ $(git -C "$repo_root" ls-files --error-unmatch "$tracked") != "$tracked" ]]; then
        printf 'formal source file is not tracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

. "$python_runtime_contract"
semtalk_require_formal_venv_python "$python_bin" semtalk

cd "$repo_root"
"$python_bin" "$partition_contract" create-run-root \
    --path "$run_root" --subdirectory logs --subdirectory candidates \
    >/dev/null

artifact_fields() {
    local path=$1 payload_flag=${2:-false}
    local -a command=(
        "$python_bin" "$partition_contract" artifact-fields
        --artifact-path "$path"
    )
    if [[ $payload_flag == true ]]; then
        command+=(--payload-receipt)
    fi
    mapfile -d '' -t artifact_result < <("${command[@]}")
    if [[ ${#artifact_result[@]} -ne 3 || \
          ! ${artifact_result[0]} =~ ^[0-9a-f]{64}$ ]]; then
        printf 'cannot pin output artifact: %s\n' "$path" >&2
        exit 1
    fi
}

evaluator_bundle=$run_root/diffsheg-evaluator-bundle.json
"$python_bin" "$partition_contract" evaluator-preflight \
    --paspa-root "$paspa_root" --diffsheg-root "$diffsheg_root" \
    --output-json "$evaluator_bundle" \
    >"$run_root/logs/evaluator-preflight.log" 2>&1
artifact_fields "$evaluator_bundle" true
evaluator_bundle_sha=${artifact_result[0]}
evaluator_bundle_payload=${artifact_result[2]}

preflight=$run_root/common-preflight.json
"$python_bin" "$inference" prepare --split val \
    --candidate-manifest "$base_candidate_manifest" \
    --expected-candidate-manifest-sha256 \
        "$expected_base_candidate_manifest_sha256" \
    --candidate-status "$base_status_json" \
    --expected-candidate-status-sha256 \
        "$expected_base_formal_status_sha256" \
    --frozen-inputs "$base_frozen_inputs_json" \
    --expected-frozen-inputs-sha256 \
        "$expected_base_frozen_inputs_sha256" \
    --val-inputs "$val_inputs_json" \
    --expected-val-inputs-sha256 "$expected_val_inputs_sha256" \
    --pipeline "$pipeline_json" \
    --expected-pipeline-sha256 "$expected_pipeline_sha256" \
    --output "$preflight" >"$run_root/logs/common-preflight.log" 2>&1
artifact_fields "$preflight" true
preflight_sha=${artifact_result[0]}
preflight_payload=${artifact_result[2]}

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
    local pid=$1 stat_line stat_tail
    local -a stat_fields
    [[ -r /proc/$pid/stat && -r /proc/$pid/cmdline ]] || return 1
    stat_line=$(<"/proc/$pid/stat") || return 1
    [[ $stat_line == *") "* ]] || return 1
    stat_tail=${stat_line##*) }
    read -r -a stat_fields <<<"$stat_tail"
    ((${#stat_fields[@]} >= 20)) || return 1
    PROC_STATE=${stat_fields[0]}
    PROC_PPID=${stat_fields[1]}
    PROC_STARTTIME=${stat_fields[19]}
    PROC_CMDLINE_SHA256=$(sha256sum "/proc/$pid/cmdline" | awk '{print $1}') \
        || return 1
}

capture_child_identity() {
    local pid=$1 expected_ppid=$2 expected_sha=$3 attempt
    for ((attempt = 0; attempt < 500; attempt++)); do
        if ! read_proc_identity "$pid"; then
            printf 'child %s exited before exact registration\n' "$pid" >&2
            return 1
        fi
        if [[ $PROC_PPID != "$expected_ppid" || $PROC_STATE == Z ]]; then
            printf 'child %s ancestry/state changed during registration\n' "$pid" >&2
            return 1
        fi
        if [[ $PROC_CMDLINE_SHA256 == "$expected_sha" ]]; then
            child_expected_ppid[$pid]=$expected_ppid
            child_starttime[$pid]=$PROC_STARTTIME
            child_cmdline_sha256[$pid]=$PROC_CMDLINE_SHA256
            active_children[$pid]=1
            return 0
        fi
        sleep 0.02
    done
    printf 'child %s never reached its exact full argv\n' "$pid" >&2
    return 1
}

validate_child_identity() {
    local pid=$1
    if ! read_proc_identity "$pid"; then
        return 2
    fi
    if [[ $PROC_STATE == Z ]]; then
        return 3
    fi
    if [[ $PROC_PPID != "${child_expected_ppid[$pid]}" || \
          $PROC_STARTTIME != "${child_starttime[$pid]}" || \
          $PROC_CMDLINE_SHA256 != "${child_cmdline_sha256[$pid]}" ]]; then
        printf 'refusing TERM for changed child identity pid=%s\n' "$pid" >&2
        return 1
    fi
}

terminate_uncaptured_child() {
    local pid=$1 expected_ppid=$2 previous_start= previous_sha= attempt
    for ((attempt = 0; attempt < 500; attempt++)); do
        if ! read_proc_identity "$pid" || [[ $PROC_STATE == Z ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        if [[ $PROC_PPID != "$expected_ppid" ]]; then
            printf 'refusing TERM for changed pending ancestry pid=%s\n' "$pid" >&2
            return 1
        fi
        if [[ -n $previous_start && $PROC_STARTTIME != "$previous_start" ]]; then
            printf 'refusing TERM for reused pending PID=%s\n' "$pid" >&2
            return 1
        fi
        if [[ $PROC_STARTTIME == "$previous_start" && \
              $PROC_CMDLINE_SHA256 == "$previous_sha" ]]; then
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        previous_start=$PROC_STARTTIME
        previous_sha=$PROC_CMDLINE_SHA256
        sleep 0.02
    done
    printf 'pending child argv never stabilized pid=%s\n' "$pid" >&2
    return 1
}

cleanup_pending_child() {
    local pid=${pending_pid:-}
    [[ -n $pid ]] || return 0
    if [[ -n ${active_children[$pid]+present} ]]; then
        pending_pid=
        pending_expected_ppid=
        return 0
    fi
    [[ -n ${pending_expected_ppid:-} ]] || return 1
    terminate_uncaptured_child "$pid" "$pending_expected_ppid"
    pending_pid=
    pending_expected_ppid=
}

terminate_children() {
    local pid identity_rc cleanup_rc=0
    local -a snapshot=("${!active_children[@]}") term_sent=()
    for pid in "${snapshot[@]}"; do
        if validate_child_identity "$pid"; then
            kill -TERM "$pid"
            term_sent+=("$pid")
            continue
        fi
        identity_rc=$?
        if [[ $identity_rc -eq 2 || $identity_rc -eq 3 ]]; then
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
    local rc=$1 cleanup_rc=0
    if [[ $launch_registration_in_progress == true && -z ${pending_pid:-} ]]; then
        pending_signal_rc=$rc
        return
    fi
    trap - EXIT INT TERM
    cleanup_pending_child || cleanup_rc=$?
    terminate_children || cleanup_rc=$?
    if [[ $cleanup_rc -ne 0 ]]; then
        printf 'signal cleanup failed rc=%s\n' "$cleanup_rc" >&2
    fi
    exit "$rc"
}

on_exit() {
    local rc=$? cleanup_rc=0
    trap - EXIT INT TERM
    cleanup_pending_child || cleanup_rc=$?
    terminate_children || cleanup_rc=$?
    if [[ $rc -eq 0 && $cleanup_rc -ne 0 ]]; then
        rc=$cleanup_rc
    fi
    exit "$rc"
}

trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

launch_registered_gpu_child() {
    local log_path=$1
    shift
    local -a argv=("$@")
    local expected_sha
    expected_sha=$(sha256_argv "${argv[@]}")
    launch_registration_in_progress=true
    pending_expected_ppid=$launcher_pid
    "${argv[@]}" >"$log_path" 2>&1 &
    pending_pid=$!
    launch_registration_in_progress=false
    if ((pending_signal_rc != 0)); then
        on_signal "$pending_signal_rc"
    fi
    capture_child_identity "$pending_pid" "$pending_expected_ppid" "$expected_sha"
    LAST_CHILD_PID=$pending_pid
    pending_pid=
    pending_expected_ppid=
}

wait_registered() {
    local pid=$1 rc
    set +e
    wait "$pid"
    rc=$?
    set -e
    unset 'active_children[$pid]'
    return "$rc"
}

mapfile -d '' -t epochs < <(
    "$python_bin" "$partition_contract" partition-epochs \
        --partition-id "$partition_id" --partition-count "$partition_count"
)
if [[ ${#epochs[@]} -ne 11 ]]; then
    printf 'static partition is not exactly eleven candidates\n' >&2
    exit 1
fi

measurement_paths=()
measurement_shas=()
for epoch in "${epochs[@]}"; do
    candidate_root=$run_root/candidates/e$epoch
    mkdir "$candidate_root"
    shard_pids=()
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
    shard_failure=0
    for pid in "${shard_pids[@]}"; do
        if ! wait_registered "$pid"; then
            shard_failure=1
        fi
    done
    if [[ $shard_failure -ne 0 ]]; then
        printf 'eight-shard validation failed for e%s\n' "$epoch" >&2
        exit 1
    fi

    "$python_bin" "$inference" finalize --split val \
        --preflight "$preflight" \
        --expected-preflight-sha256 "$preflight_sha" \
        --epoch "$epoch" --output-root "$candidate_root" --num-shards 8 \
        >"$run_root/logs/e${epoch}-finalize.log" 2>&1
    lineage=$candidate_root/final/val-inference-lineage.json
    clip_manifest=$candidate_root/final/diffsheg_eval_clip_ids.txt
    artifact_fields "$lineage" true
    lineage_sha=${artifact_result[0]}
    artifact_fields "$clip_manifest" false
    clip_manifest_sha=${artifact_result[0]}

    report=$candidate_root/diffsheg-val-fgd.json
    metric_command=(
        "$python_bin" "$evaluator"
        --pred-dir "$candidate_root/final/predictions/val"
        --gt-dir "$candidate_root/final/ground-truth/val"
        --clip-manifest "$clip_manifest"
        --clip-manifest-sha256 "$clip_manifest_sha"
        --paspa-root "$paspa_root" --diffsheg-root "$diffsheg_root"
        --device cuda:0 --batch-size "$diffsheg_batch_size"
        --output "$report"
    )
    launch_registered_gpu_child \
        "$run_root/logs/e${epoch}-diffsheg-fgd.log" \
        "${metric_command[@]}"
    if ! wait_registered "$LAST_CHILD_PID"; then
        printf 'DiffSHEG validation FGD failed for e%s\n' "$epoch" >&2
        exit 1
    fi
    artifact_fields "$report" false
    report_sha=${artifact_result[0]}

    measurement=$candidate_root/diffsheg-val-measurement.json
    "$python_bin" "$measurement_producer" \
        --epoch "$epoch" \
        --base-candidate-manifest "$base_candidate_manifest" \
        --expected-base-candidate-manifest-sha256 \
            "$expected_base_candidate_manifest_sha256" \
        --base-status-json "$base_status_json" \
        --expected-base-formal-status-sha256 \
            "$expected_base_formal_status_sha256" \
        --base-frozen-inputs-json "$base_frozen_inputs_json" \
        --expected-base-frozen-inputs-sha256 \
            "$expected_base_frozen_inputs_sha256" \
        --val-inputs-json "$val_inputs_json" \
        --expected-val-inputs-sha256 "$expected_val_inputs_sha256" \
        --pipeline-json "$pipeline_json" \
        --expected-pipeline-sha256 "$expected_pipeline_sha256" \
        --inference-lineage-json "$lineage" \
        --expected-inference-lineage-sha256 "$lineage_sha" \
        --diffsheg-report-json "$report" \
        --expected-diffsheg-report-sha256 "$report_sha" \
        --output-json "$measurement" \
        >"$run_root/logs/e${epoch}-measurement.log" 2>&1
    artifact_fields "$measurement" true
    measurement_paths+=("$measurement")
    measurement_shas+=("${artifact_result[0]}")
done

partition_receipt=$run_root/partition-receipt.json
seal_command=(
    "$python_bin" "$partition_contract" seal-partition
    --partition-id "$partition_id" --partition-count "$partition_count"
    --formal-host "$formal_host" --run-root "$run_root"
    --source-commit "$source_commit" --source-tree "$source_tree"
    --base-candidate-manifest "$base_candidate_manifest"
    --expected-base-candidate-manifest-sha256
        "$expected_base_candidate_manifest_sha256"
    --base-status-json "$base_status_json"
    --expected-base-formal-status-sha256
        "$expected_base_formal_status_sha256"
    --base-frozen-inputs-json "$base_frozen_inputs_json"
    --expected-base-frozen-inputs-sha256
        "$expected_base_frozen_inputs_sha256"
    --val-inputs-json "$val_inputs_json"
    --expected-val-inputs-sha256 "$expected_val_inputs_sha256"
    --pipeline-json "$pipeline_json"
    --expected-pipeline-sha256 "$expected_pipeline_sha256"
    --preflight-json "$preflight"
    --expected-preflight-sha256 "$preflight_sha"
    --expected-preflight-payload-sha256 "$preflight_payload"
    --evaluator-bundle-json "$evaluator_bundle"
    --expected-evaluator-bundle-sha256 "$evaluator_bundle_sha"
    --expected-evaluator-bundle-payload-sha256 "$evaluator_bundle_payload"
)
for ((index = 0; index < ${#measurement_paths[@]}; index++)); do
    seal_command+=(
        --measurement-json "${measurement_paths[$index]}"
        --expected-measurement-sha256 "${measurement_shas[$index]}"
    )
done
seal_command+=(--output-json "$partition_receipt")
"${seal_command[@]}" >"$run_root/logs/partition-seal.log" 2>&1

trap - EXIT INT TERM
artifact_fields "$partition_receipt" true
printf '%s\0%s\0%s\0' \
    "$partition_receipt" "${artifact_result[0]}" "${artifact_result[2]}"
