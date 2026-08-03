#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Consume exactly one create-new live Base candidate authority.  The direct
# parent must be the all-GPU guarded runner.  Eight exact argv-registered
# inference shards use cuda:0..7; after their exact-once finalize, one
# registered cuda:0 DiffSHEG process publishes the validation-only FGD report.

usage() {
    printf '%s\n' \
        "Usage: $0 --repo-root PATH --python PATH --work-authority PATH --expected-work-authority-sha256 SHA --run-root NEW_PATH --source-commit OID --source-tree OID --paspa-root PATH --diffsheg-root PATH --seed INT --diffsheg-batch-size INT [--recovery-authority PATH --expected-recovery-authority-sha256 SHA --recovery-claim PATH --expected-recovery-claim-sha256 SHA]"
}

if ((BASH_VERSINFO[0] < 5)); then
    printf 'bash >= 5 is required\n' >&2
    exit 1
fi

declare -A seen_options=()
repo_root=
python_bin=
work_authority=
expected_work_authority_sha256=
run_root=
source_commit=
source_tree=
paspa_root=
diffsheg_root=
seed=
diffsheg_batch_size=
recovery_authority=
expected_recovery_authority_sha256=
recovery_claim=
expected_recovery_claim_sha256=

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
        --work-authority) set_option "$1" work_authority "$2" ;;
        --expected-work-authority-sha256)
            set_option "$1" expected_work_authority_sha256 "$2" ;;
        --run-root) set_option "$1" run_root "$2" ;;
        --source-commit) set_option "$1" source_commit "$2" ;;
        --source-tree) set_option "$1" source_tree "$2" ;;
        --paspa-root) set_option "$1" paspa_root "$2" ;;
        --diffsheg-root) set_option "$1" diffsheg_root "$2" ;;
        --seed) set_option "$1" seed "$2" ;;
        --diffsheg-batch-size)
            set_option "$1" diffsheg_batch_size "$2" ;;
        --recovery-authority)
            set_option "$1" recovery_authority "$2" ;;
        --expected-recovery-authority-sha256)
            set_option "$1" expected_recovery_authority_sha256 "$2" ;;
        --recovery-claim)
            set_option "$1" recovery_claim "$2" ;;
        --expected-recovery-claim-sha256)
            set_option "$1" expected_recovery_claim_sha256 "$2" ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift 2
done

for required in \
    repo_root python_bin work_authority expected_work_authority_sha256 \
    run_root source_commit source_tree paspa_root diffsheg_root seed \
    diffsheg_batch_size; do
    if [[ -z ${!required} ]]; then
        printf 'missing required option value: %s\n' "$required" >&2
        exit 2
    fi
done
recovery_option_count=0
for optional in \
    recovery_authority expected_recovery_authority_sha256 recovery_claim \
    expected_recovery_claim_sha256; do
    [[ -n ${!optional} ]] && ((recovery_option_count += 1))
done
if ((recovery_option_count != 0 && recovery_option_count != 4)); then
    printf 'recovery options must be supplied all-or-none\n' >&2
    exit 2
fi
if [[ ! $expected_work_authority_sha256 =~ ^[0-9a-f]{64}$ || \
      ! $source_commit =~ ^[0-9a-f]{40}$ || \
      ! $source_tree =~ ^[0-9a-f]{40}$ || \
      ! $seed =~ ^[0-9]+$ || \
      ! $diffsheg_batch_size =~ ^[1-9][0-9]*$ ]]; then
    printf 'invalid SHA, Git OID, seed, or DiffSHEG batch size\n' >&2
    exit 2
fi
if ((recovery_option_count == 4)) && \
   [[ ! $expected_recovery_authority_sha256 =~ ^[0-9a-f]{64}$ || \
      ! $expected_recovery_claim_sha256 =~ ^[0-9a-f]{64}$ ]]; then
    printf 'invalid recovery SHA\n' >&2
    exit 2
fi
for path_value in \
    "$repo_root" "$python_bin" "$work_authority" "$run_root" \
    "$paspa_root" "$diffsheg_root"; do
    [[ $path_value == /* ]] || {
        printf 'every live validation path must be absolute\n' >&2
        exit 2
    }
done
if ((recovery_option_count == 4)); then
    for path_value in "$recovery_authority" "$recovery_claim"; do
        [[ $path_value == /* ]] || {
            printf 'every recovery path must be absolute\n' >&2
            exit 2
        }
    done
fi

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
raw_work_authority=$work_authority
work_authority=$(realpath -e -- "$work_authority")
if [[ $work_authority != "$raw_work_authority" || \
      ! -f $work_authority || -L $raw_work_authority ]]; then
    printf 'work authority must be one canonical regular file\n' >&2
    exit 1
fi
if ((recovery_option_count == 4)); then
    raw_recovery_authority=$recovery_authority
    recovery_authority=$(realpath -e -- "$recovery_authority")
    raw_recovery_claim=$recovery_claim
    recovery_claim=$(realpath -e -- "$recovery_claim")
    if [[ $recovery_authority != "$raw_recovery_authority" || \
          ! -f $recovery_authority || -L $raw_recovery_authority || \
          $recovery_claim != "$raw_recovery_claim" || \
          ! -f $recovery_claim || -L $raw_recovery_claim ]]; then
        printf 'recovery authority/claim must be canonical regular files\n' >&2
        exit 1
    fi
fi
run_parent=$(realpath -e -- "$(dirname -- "$run_root")")
if [[ $run_root != "$run_parent/$(basename -- "$run_root")" || \
      -e $run_root || -L $run_root ]]; then
    printf 'run root must be one new canonical path\n' >&2
    exit 1
fi
case "$run_root/" in
    "$repo_root/"*)
        printf 'run root must remain outside the source checkout\n' >&2
        exit 1
        ;;
esac

launcher_name=run_base_live_val_8shard.sh
if [[ -L ${BASH_SOURCE[0]} ]]; then
    printf 'launcher must not be a symlink\n' >&2
    exit 1
fi
launcher_path=$(realpath -e -- "${BASH_SOURCE[0]}")
launcher_dir=$repo_root/scripts/show_base
if [[ $launcher_path != "$launcher_dir/$launcher_name" ]]; then
    printf 'launcher is not the tracked entrypoint in the pinned source\n' >&2
    exit 1
fi
bridge=$launcher_dir/base_live_val_consumer_bridge.py
control_evaluator=$launcher_dir/evaluate_diffsheg_val_fgd.py
partition_contract=$launcher_dir/base_diffsheg_val_partition_contract.py
guard_contract=$launcher_dir/guarded_runner_contract.sh
python_runtime_contract=$launcher_dir/formal_python_runtime_contract.sh
for required_path in \
    "$bridge" "$control_evaluator" "$partition_contract" "$guard_contract" \
    "$python_runtime_contract"; do
    if [[ ! -f $required_path || -L $required_path ]]; then
        printf 'required tracked source is unavailable: %s\n' "$required_path" >&2
        exit 1
    fi
done

. "$guard_contract"
semtalk_require_exact_guarded_runner_all_gpus

mapfile -t git_remotes < <(git -C "$repo_root" remote)
if [[ ${#git_remotes[@]} -ne 1 || ${git_remotes[0]} != origin || \
      $(git -C "$repo_root" remote get-url origin) != \
          git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" remote get-url --push origin) != \
          git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" rev-parse HEAD) != "$source_commit" || \
      $(git -C "$repo_root" rev-parse 'HEAD^{tree}') != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'source must be exact clean detached zero-branch SemTalk\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/base_live_val_consumer_bridge.py \
    scripts/show_base/run_base_live_val_8shard.sh \
    scripts/show_base/guarded_runner_contract.sh \
    scripts/show_base/formal_python_runtime_contract.sh \
    scripts/show_base/base_diffsheg_val_partition_contract.py \
    scripts/show_base/run_base_val_inference.py \
    scripts/show_base/semtalk_base_inference_core.py \
    scripts/show_base/evaluate_diffsheg_val_fgd.py \
    scripts/show_base/produce_base_val_measurement.py \
    scripts/show_base/select_base_official_adapt_long.py \
    scripts/show_base/base_long_val_contract.py; do
    if [[ $(git -C "$repo_root" ls-files --error-unmatch "$tracked") != "$tracked" ]]; then
        printf 'live validation source file is not tracked: %s\n' "$tracked" >&2
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

preflight=$run_root/work-preflight.json
prepare_command=("$python_bin" "$bridge" prepare \
    --work-authority "$work_authority" \
    --expected-work-authority-sha256 "$expected_work_authority_sha256" \
    --run-root "$run_root" --output "$preflight")
if ((recovery_option_count == 4)); then
    prepare_command+=(
        --recovery-authority "$recovery_authority"
        --expected-recovery-authority-sha256 \
            "$expected_recovery_authority_sha256"
        --recovery-claim "$recovery_claim"
        --expected-recovery-claim-sha256 "$expected_recovery_claim_sha256"
    )
fi
"${prepare_command[@]}" >"$run_root/logs/work-preflight.log" 2>&1
artifact_fields "$preflight" true
preflight_sha=${artifact_result[0]}

"$python_bin" "$bridge" inspect \
    --preflight "$preflight" \
    --expected-preflight-sha256 "$preflight_sha" \
    >"$run_root/logs/work-inspect.json"
mapfile -d '' -t inspected_work < <(
"$python_bin" - "$run_root/logs/work-inspect.json" <<'PY'
import json
from pathlib import Path
import re
import sys

def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")

def exact_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value

value = json.loads(
    Path(sys.argv[1]).read_bytes(),
    object_pairs_hook=exact_object,
    parse_constant=reject_constant,
)
if type(value) is not dict or set(value) != {
    "status", "split", "test_visible", "selection_eligible", "epoch",
    "preflight", "frozen_evidence_evaluator",
}:
    raise SystemExit("inspect result schema changed")
epoch = value.get("epoch")
preflight = value.get("preflight")
adapter = value.get("frozen_evidence_evaluator")
if (
    value.get("status") != "ready"
    or value.get("split") != "val"
    or value.get("test_visible") is not False
    or value.get("selection_eligible") is not False
    or type(epoch) is not int
    or epoch < 1
    or type(preflight) is not dict
    or set(preflight) != {
        "path", "sha256", "bytes", "receipt_payload_sha256"
    }
    or type(adapter) is not dict
    or set(adapter) != {
        "path", "sha256", "bytes", "git_mode", "git_blob_sha1",
        "repository_root", "repository_git_head", "repository_git_tree",
        "repository_origin", "repository_clean", "repository_detached",
        "repository_local_branches_at_commit",
    }
):
    raise SystemExit("inspect result is not one exact frozen evaluator binding")
hex40 = re.compile(r"[0-9a-f]{40}\Z")
hex64 = re.compile(r"[0-9a-f]{64}\Z")
if (
    type(preflight["path"]) is not str
    or not preflight["path"].startswith("/")
    or type(preflight["sha256"]) is not str
    or hex64.fullmatch(preflight["sha256"]) is None
    or type(preflight["bytes"]) is not int
    or preflight["bytes"] < 1
    or type(preflight["receipt_payload_sha256"]) is not str
    or hex64.fullmatch(preflight["receipt_payload_sha256"]) is None
    or type(adapter["path"]) is not str
    or not adapter["path"].startswith("/")
    or type(adapter["sha256"]) is not str
    or hex64.fullmatch(adapter["sha256"]) is None
    or type(adapter["bytes"]) is not int
    or adapter["bytes"] < 1
    or adapter["git_mode"] not in {"100644", "100755"}
    or type(adapter["git_blob_sha1"]) is not str
    or hex40.fullmatch(adapter["git_blob_sha1"]) is None
    or type(adapter["repository_root"]) is not str
    or not adapter["repository_root"].startswith("/")
    or type(adapter["repository_git_head"]) is not str
    or hex40.fullmatch(adapter["repository_git_head"]) is None
    or type(adapter["repository_git_tree"]) is not str
    or hex40.fullmatch(adapter["repository_git_tree"]) is None
    or adapter["repository_origin"]
    != "git@github.com:Xiangyue-Zhang/SemTalk.git"
    or adapter["repository_clean"] is not True
    or adapter["repository_detached"] is not True
    or adapter["repository_local_branches_at_commit"] != []
):
    raise SystemExit("frozen evaluator identity is invalid")
fields = (
    str(epoch), adapter["path"], adapter["sha256"], str(adapter["bytes"]),
    adapter["git_mode"], adapter["git_blob_sha1"],
    adapter["repository_root"], adapter["repository_git_head"],
    adapter["repository_git_tree"], adapter["repository_origin"],
)
sys.stdout.buffer.write(b"\0".join(item.encode("utf-8") for item in fields) + b"\0")
PY
)
if [[ ${#inspected_work[@]} -ne 10 ]]; then
    printf 'bridge did not expose one exact frozen evaluator binding\n' >&2
    exit 1
fi
epoch=${inspected_work[0]}
raw_evaluator=${inspected_work[1]}
evaluator_sha256=${inspected_work[2]}
evaluator_bytes=${inspected_work[3]}
evaluator_git_mode=${inspected_work[4]}
evaluator_git_blob=${inspected_work[5]}
raw_evaluator_root=${inspected_work[6]}
evaluator_git_head=${inspected_work[7]}
evaluator_git_tree=${inspected_work[8]}
evaluator_origin=${inspected_work[9]}
evaluator=$(realpath -e -- "$raw_evaluator")
evaluator_root=$(realpath -e -- "$raw_evaluator_root")
evaluator_relative=scripts/show_base/evaluate_diffsheg_val_fgd.py
if [[ ! $epoch =~ ^[1-9][0-9]*$ || \
      $evaluator != "$raw_evaluator" || ! -f $evaluator || \
      -L $raw_evaluator || $evaluator_root != "$raw_evaluator_root" || \
      ! -d $evaluator_root || -L $raw_evaluator_root || \
      $evaluator != "$evaluator_root/$evaluator_relative" || \
      $(sha256sum "$evaluator" | awk '{print $1}') != "$evaluator_sha256" || \
      $(stat -c '%s' "$evaluator") != "$evaluator_bytes" || \
      $evaluator_origin != git@github.com:Xiangyue-Zhang/SemTalk.git ]]; then
    printf 'frozen evidence evaluator file binding changed\n' >&2
    exit 1
fi
mapfile -t evaluator_remotes < <(git -C "$evaluator_root" remote)
set +e
evaluator_symbolic_ref=$(git -C "$evaluator_root" \
    symbolic-ref -q --short HEAD 2>/dev/null)
evaluator_symbolic_rc=$?
set -e
expected_evaluator_tree_entry="$evaluator_git_mode blob $evaluator_git_blob"$'\t'"$evaluator_relative"
if [[ ${#evaluator_remotes[@]} -ne 1 || \
      ${evaluator_remotes[0]} != origin || \
      $(git -C "$evaluator_root" remote get-url origin) != "$evaluator_origin" || \
      $(git -C "$evaluator_root" remote get-url --push origin) != "$evaluator_origin" || \
      $(git -C "$evaluator_root" rev-parse HEAD) != "$evaluator_git_head" || \
      $(git -C "$evaluator_root" rev-parse 'HEAD^{tree}') != "$evaluator_git_tree" || \
      $(git -C "$evaluator_root" ls-files --error-unmatch \
          "$evaluator_relative") != "$evaluator_relative" || \
      $(git -C "$evaluator_root" ls-tree HEAD -- \
          "$evaluator_relative") != "$expected_evaluator_tree_entry" || \
      -n "$(git -C "$evaluator_root" status --porcelain=v1 \
          --untracked-files=all)" || $evaluator_symbolic_rc -ne 1 || \
      -n $evaluator_symbolic_ref || \
      -n "$(git -C "$evaluator_root" for-each-ref \
          --format='%(refname)' refs/heads)" ]]; then
    printf 'frozen evidence evaluator repository identity changed\n' >&2
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

candidate_root=$run_root/candidates/e$epoch
mkdir "$candidate_root"
shard_pids=()
for shard_id in 0 1 2 3 4 5 6 7; do
    shard_command=(
        "$python_bin" "$bridge" shard --split val
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
    printf 'eight-shard live validation failed for e%s\n' "$epoch" >&2
    exit 1
fi

"$python_bin" "$bridge" finalize --split val \
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
    printf 'DiffSHEG live validation FGD failed for e%s\n' "$epoch" >&2
    exit 1
fi
artifact_fields "$report" false
report_sha=${artifact_result[0]}

measurement=$candidate_root/live-measurement.json
"$python_bin" "$bridge" complete \
    --preflight "$preflight" \
    --expected-preflight-sha256 "$preflight_sha" \
    --inference-lineage "$lineage" \
    --expected-inference-lineage-sha256 "$lineage_sha" \
    --diffsheg-report "$report" \
    --expected-diffsheg-report-sha256 "$report_sha" \
    --output "$measurement" \
    >"$run_root/logs/e${epoch}-live-measurement.log" 2>&1
artifact_fields "$measurement" true
measurement_sha=${artifact_result[0]}
measurement_payload=${artifact_result[2]}

trap - EXIT INT TERM
printf '%s\0%s\0%s\0' \
    "$measurement" "$measurement_sha" "$measurement_payload"
