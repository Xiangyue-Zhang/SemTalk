#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# This launcher must itself run as the child of /tmp/globaldiff_guarded_runner.py
# with physical GPUs 0..7 reserved.  It assigns one isolated GPU to each of the
# five independent representation models.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON REP_LMDB REP_SUMMARY LINEAGE ASSET_ROOT OUTPUT_ROOT RUN_ID PARITY_BUNDLE PARITY_SHA256 LOWER_TARGET_CACHE LOWER_TARGET_MANIFEST LOWER_TARGET_MANIFEST_SHA256 LOWER_TARGET_CHECKER LOWER_TARGET_CHECKER_SHA256 [--resume]"
}

if [[ $# -ne 15 && $# -ne 16 ]]; then
    usage
    exit 2
fi

repo_root=$1
python_bin=$2
rep_lmdb=$3
rep_summary=$4
lineage=$5
asset_root=$6
output_root=$7
run_id=$8
parity_bundle=$9
parity_sha256=${10}
lower_target_cache=${11}
lower_target_manifest=${12}
lower_target_manifest_sha256=${13}
lower_target_checker=${14}
lower_target_checker_sha256=${15}
resume_mode=false
formal_smplx_sha256=bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74
if [[ $# -eq 16 ]]; then
    if [[ ${16} != "--resume" ]]; then
        usage
        exit 2
    fi
    resume_mode=true
fi

for required in "$repo_root/show_base_train.py" "$python_bin" "$rep_summary" \
    "$lineage" "$parity_bundle" "$lower_target_manifest" \
    "$lower_target_checker"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ ! -d "$rep_lmdb" ]]; then
    printf 'missing representation LMDB: %s\n' "$rep_lmdb" >&2
    exit 1
fi
if [[ ! -d "$lower_target_cache" ]]; then
    printf 'missing lower target cache LMDB: %s\n' \
        "$lower_target_cache" >&2
    exit 1
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'unsafe run id: %s\n' "$run_id" >&2
    exit 1
fi

read -r train_samples updates_per_epoch global_foot_fastpath < <(
    "$python_bin" - "$rep_summary" "$rep_lmdb" "$lineage" \
        "$parity_bundle" "$parity_sha256" "$asset_root" \
        "$lower_target_cache" "$lower_target_manifest" \
        "$lower_target_manifest_sha256" "$lower_target_checker" \
        "$lower_target_checker_sha256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def require_exact_int(value, label):
    if type(value) is not int:
        raise SystemExit(f"{label} must be an exact integer")
    return value

summary_path = Path(sys.argv[1]).resolve()
lmdb_path = Path(sys.argv[2]).resolve()
lineage_path = Path(sys.argv[3]).resolve()
parity_path = Path(sys.argv[4]).resolve()
expected_parity_sha = sys.argv[5]
asset_root = Path(sys.argv[6]).resolve()
lower_cache_input = Path(sys.argv[7])
lower_manifest_input = Path(sys.argv[8])
expected_lower_manifest_sha = sys.argv[9]
lower_checker_input = Path(sys.argv[10])
expected_lower_checker_sha = sys.argv[11]
if summary_path != lineage_path:
    raise SystemExit(
        "representation lineage must be the exact representation summary"
    )
summary = json.loads(summary_path.read_text())
summary_format = summary.get("format")
if (
    summary.get("status") != "complete"
    or summary_format != "semtalk_show_representation_lmdb_v2_global_foot"
):
    raise SystemExit("representation summary is not complete")
fastpath = 1
if summary.get("protocol", {}).get(
    "global_foot_fastpath"
) != {
    "enabled": True,
    "contract": "semtalk_show_global_foot_fastpath_v1",
    "field": "lower_foot_local",
    "shape": [64, 4, 3],
    "dtype": "float32",
    "activation_env": "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH=1",
}:
    raise SystemExit("invalid Global-foot fastpath receipt")
if (
    len(expected_parity_sha) != 64
    or any(char not in "0123456789abcdef" for char in expected_parity_sha)
):
    raise SystemExit("invalid expected parity SHA-256")
parity_digest = hashlib.sha256(parity_path.read_bytes()).hexdigest()
if parity_digest != expected_parity_sha:
    raise SystemExit("Global-foot parity bundle SHA mismatch")
parity = json.loads(parity_path.read_text())
expected_speakers = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
def exact_speaker_map(value, label):
    if not isinstance(value, dict) or set(value) != set(expected_speakers):
        raise SystemExit(f"{label} has invalid keys")
    for speaker, expected_id in expected_speakers.items():
        speaker_id = value[speaker]
        if type(speaker_id) is not int or speaker_id != expected_id:
            raise SystemExit(f"{label}.{speaker} is not the exact speaker ID")
    return value

if (
    parity.get("format") != "semtalk_show_global_foot_parity_suite_v1"
    or parity.get("status") != "pass"
    or parity.get("contract") != "semtalk_show_global_foot_fastpath_v1"
    or exact_speaker_map(parity.get("speakers"), "parity speakers")
    != expected_speakers
    or parity.get("canonical_receipt") != summary.get("canonical_receipt")
):
    raise SystemExit("invalid Global-foot parity bundle")
summary_source = summary.get("source_receipt")
parity_source = parity.get("source_receipt")
if (
    not isinstance(summary_source, dict)
    or not isinstance(parity_source, dict)
    or {
        key: parity_source.get(key) for key in ("origin", "commit", "tree")
    }
    != {
        key: summary_source.get(key) for key in ("origin", "commit", "tree")
    }
):
    raise SystemExit("parity/representation source receipt mismatch")
reports = parity.get("reports")
report_speakers = set()
if isinstance(reports, list):
    for record in reports:
        if not isinstance(record, dict):
            break
        speaker_id = record.get("speaker_id")
        if type(speaker_id) is not int:
            raise SystemExit("parity report speaker_id is not an exact integer")
        report_speakers.add((record.get("speaker"), speaker_id))
if (
    not isinstance(reports, list)
    or len(reports) != 4
    or report_speakers != set(expected_speakers.items())
):
    raise SystemExit("parity suite does not cover all four SHOW speakers")
for record in reports:
    report_path = Path(record["report"]).resolve()
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    report_payload = json.loads(report_path.read_text())
    if (
        report_sha != record.get("report_sha256")
        or report_payload != record.get("payload")
        or report_payload.get("status") != "pass"
        or report_payload.get("contract")
        != "semtalk_show_global_foot_fastpath_v1"
    ):
        raise SystemExit(f"invalid parity report: {report_path}")
smplx_asset = (
    asset_root
    / "smplx_models"
    / "smplx"
    / "SMPLX_NEUTRAL_2020.npz"
)
smplx_sha = hashlib.sha256(smplx_asset.read_bytes()).hexdigest()
if smplx_sha != parity.get("smplx_asset_sha256"):
    raise SystemExit("parity/asset SMPL-X SHA mismatch")
if Path(summary["lmdb"]).resolve() != lmdb_path:
    raise SystemExit("representation LMDB path mismatch")
digest_state = hashlib.sha256()
with (lmdb_path / "data.mdb").open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest_state.update(chunk)
digest = digest_state.hexdigest()
if digest != summary["data_mdb_sha256"]:
    raise SystemExit("representation data.mdb SHA mismatch")

def require_sha256(value, label):
    if (
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SystemExit(f"{label} must be a lowercase SHA-256")
    return value

def sha256_file(path):
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()

if lower_cache_input.is_symlink():
    raise SystemExit("lower target cache LMDB must not be a symlink")
lower_cache = lower_cache_input.resolve()
if not lower_cache.is_dir():
    raise SystemExit("lower target cache LMDB is missing")
lower_data = lower_cache / "data.mdb"
lower_lock = lower_cache / "lock.mdb"
for artifact in (lower_data, lower_lock):
    if artifact.is_symlink() or not artifact.is_file():
        raise SystemExit(f"invalid lower target cache artifact: {artifact}")
for receipt_path, expected_sha, label in (
    (
        lower_manifest_input,
        expected_lower_manifest_sha,
        "lower target cache manifest",
    ),
    (
        lower_checker_input,
        expected_lower_checker_sha,
        "lower target cache checker",
    ),
):
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SystemExit(f"{label} must be a regular non-symlink file")
    actual_sha = sha256_file(receipt_path)
    if actual_sha != require_sha256(expected_sha, f"{label} SHA-256"):
        raise SystemExit(f"{label} SHA mismatch")
lower_manifest = json.loads(lower_manifest_input.read_text())
lower_checker = json.loads(lower_checker_input.read_text())
if not isinstance(lower_manifest, dict) or not isinstance(lower_checker, dict):
    raise SystemExit("lower target cache receipts must be JSON objects")
lower_manifest_lmdb = lower_manifest.get("lmdb")
if (
    lower_manifest.get("format")
    != "semtalk_show_lower_target_joints_raw_lmdb_v1"
    or lower_manifest.get("status") != "complete"
    or not isinstance(lower_manifest_lmdb, dict)
    or Path(str(lower_manifest_lmdb.get("path", ""))).resolve()
    != lower_cache
    or lower_manifest_lmdb.get("data_mdb_sha256")
    != sha256_file(lower_data)
    or lower_manifest_lmdb.get("lock_mdb_sha256")
    != sha256_file(lower_lock)
):
    raise SystemExit("lower target cache manifest/LMDB binding is invalid")
if (
    lower_checker.get("format")
    != "semtalk_show_lower_target_joints_checker_v1"
    or lower_checker.get("status") != "complete"
    or lower_checker.get("cache_manifest_sha256")
    != expected_lower_manifest_sha
    or Path(str(lower_checker.get("cache_path", ""))).resolve()
    != lower_cache
    or lower_checker.get("cache_data_mdb_sha256")
    != lower_manifest_lmdb["data_mdb_sha256"]
    or lower_checker.get("entry_aggregate_sha256")
    != lower_manifest.get("entry_aggregate_sha256")
    or lower_checker.get("torch_equal_all") is not True
    or lower_checker.get("exact_once") is not True
    or lower_checker.get("finite") is not True
):
    raise SystemExit("lower target cache checker binding is invalid")
entries = require_exact_int(summary.get("entries"), "representation entries")
updates = entries // 64
if entries != 127_309 or updates != 1_989:
    raise SystemExit(
        f"formal representation accounting mismatch: {entries=} {updates=}"
    )
print(entries, updates, fastpath)
PY
)

mkdir -p "$output_root/logs/$run_id"
for stage in face hands upper lower global; do
    stage_dir="$output_root/$stage/custom/${run_id}_${stage}"
    if [[ "$resume_mode" == false && -e "$stage_dir" ]]; then
        printf 'refusing to reuse stage output: %s\n' "$stage" >&2
        exit 1
    fi
    if [[ "$resume_mode" == true && ! -d "$stage_dir" ]]; then
        printf 'resume stage output is missing: %s\n' "$stage_dir" >&2
        exit 1
    fi
done

declare -a child_pids=()
declare -a active_pids=()
declare -A child_name_by_pid=()
declare -A child_start_by_pid=()
declare -A child_cmd_sha_by_pid=()
pending_pid=
pending_signal_rc=0
launch_registration_in_progress=false

capture_child_identity() {
    local pid=$1
    local stage_run=$2
    "$python_bin" - "$pid" "$$" "$stage_run" <<'PY'
import hashlib
from pathlib import Path
import sys
import time

pid = int(sys.argv[1])
expected_ppid = int(sys.argv[2])
expected_run = sys.argv[3].encode()
proc = Path("/proc") / str(pid)
for _ in range(200):
    try:
        fields = (proc / "stat").read_text().split()
        command = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        raise SystemExit(f"child {pid} exited before identity capture")
    argv = [item for item in command.split(b"\0") if item]
    if (
        int(fields[3]) == expected_ppid
        and b"show_base_train.py" in argv
        and expected_run in argv
    ):
        print(fields[21], hashlib.sha256(command).hexdigest())
        raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit(f"child {pid} never reached its exact training argv")
PY
}

verify_child_identity() {
    local pid=$1
    local expected_start=$2
    local expected_cmd_sha=$3
    "$python_bin" - "$pid" "$$" "$expected_start" "$expected_cmd_sha" <<'PY'
import hashlib
from pathlib import Path
import sys

pid = int(sys.argv[1])
expected_ppid = int(sys.argv[2])
expected_start = sys.argv[3]
expected_cmd_sha = sys.argv[4]
proc = Path("/proc") / str(pid)
try:
    fields = (proc / "stat").read_text().split()
    command = (proc / "cmdline").read_bytes()
except FileNotFoundError:
    raise SystemExit(1)
valid = (
    int(fields[3]) == expected_ppid
    and fields[21] == expected_start
    and hashlib.sha256(command).hexdigest() == expected_cmd_sha
)
raise SystemExit(0 if valid else 1)
PY
}

terminate_uncaptured_child() {
    local pid=$1
    local identity snapshot_rc state start cmd_sha
    local previous_start= previous_cmd_sha=
    local attempt
    for ((attempt = 0; attempt < 500; attempt++)); do
        set +e
        identity=$(
            "$python_bin" - "$pid" "$$" <<'PY'
import hashlib
from pathlib import Path
import sys

pid = int(sys.argv[1])
expected_ppid = int(sys.argv[2])
proc = Path("/proc") / str(pid)
try:
    fields = (proc / "stat").read_text().split()
    command = (proc / "cmdline").read_bytes()
except FileNotFoundError:
    raise SystemExit(2)
if int(fields[3]) != expected_ppid:
    raise SystemExit(3)
print(fields[2], fields[21], hashlib.sha256(command).hexdigest())
PY
        )
        snapshot_rc=$?
        set -e
        if ((snapshot_rc == 2)); then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        if ((snapshot_rc != 0)); then
            printf 'refusing TERM: uncaptured child ancestry changed pid=%s\n' \
                "$pid" >&2
            return 1
        fi
        read -r state start cmd_sha <<<"$identity"
        if [[ "$state" == Z ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        if [[ -n "$previous_start" && "$start" != "$previous_start" ]]; then
            printf 'refusing TERM: pending PID starttime changed pid=%s\n' \
                "$pid" >&2
            return 1
        fi
        if [[ "$start" == "$previous_start" && \
              "$cmd_sha" == "$previous_cmd_sha" ]]; then
            kill -TERM "$pid"
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        previous_start=$start
        previous_cmd_sha=$cmd_sha
        sleep 0.02
    done
    printf 'refusing TERM: pending child argv never stabilized pid=%s\n' \
        "$pid" >&2
    return 1
}

is_registered_active_child() {
    local pid=$1
    local active_pid
    [[ -n "${child_name_by_pid[$pid]+present}" && \
       -n "${child_start_by_pid[$pid]+present}" && \
       -n "${child_cmd_sha_by_pid[$pid]+present}" ]] || return 1
    for active_pid in "${active_pids[@]:-}"; do
        if [[ "$active_pid" == "$pid" ]]; then
            return 0
        fi
    done
    return 1
}

cleanup_pending_child() {
    local pid=${pending_pid:-}
    [[ -n "$pid" ]] || return 0
    if is_registered_active_child "$pid"; then
        pending_pid=
        return 0
    fi
    if terminate_uncaptured_child "$pid"; then
        pending_pid=
        return 0
    fi
    printf 'failed to clean exact pending child pid=%s\n' "$pid" >&2
    return 1
}

terminate_children() {
    local pid cleanup_rc=0
    local -a term_sent=()
    for pid in "${active_pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null && verify_child_identity \
            "$pid" "${child_start_by_pid[$pid]}" \
            "${child_cmd_sha_by_pid[$pid]}"; then
            if kill -TERM "$pid" 2>/dev/null; then
                term_sent+=("$pid")
            else
                cleanup_rc=1
            fi
        elif kill -0 "$pid" 2>/dev/null; then
            printf 'refusing TERM: child identity changed pid=%s\n' \
                "$pid" >&2
            cleanup_rc=1
        else
            wait "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${term_sent[@]:-}"; do
        wait "$pid" 2>/dev/null || true
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
    if ((cleanup_rc != 0)); then
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
    if ((rc == 0 && cleanup_rc != 0)); then
        rc=$cleanup_rc
    fi
    exit "$rc"
}

trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

launch_stage() {
    local stage=$1
    local gpu=$2
    local port=$3
    local config=$4
    local final_name=$5
    local epochs=$6
    local stage_out="$output_root/$stage/"
    local stage_run="${run_id}_${stage}"
    local stage_dir="$stage_out/custom/$stage_run"
    local resume_args=()
    local parity_args=()
    local smplx_args=()
    local lower_cache_args=()
    local log_path="$output_root/logs/$run_id/$stage.log"
    mkdir -p "$stage_out"

    if [[ "$resume_mode" == true ]]; then
        if [[ ! -f "$stage_dir/latest_resume.pt" ]]; then
            printf 'resume checkpoint is missing: %s\n' \
                "$stage_dir/latest_resume.pt" >&2
            return 1
        fi
        resume_args=(--resume_state "$stage_dir/latest_resume.pt")
        log_path="$output_root/logs/$run_id/$stage.resume.$$.log"
    fi
    if [[ "$stage" == global ]]; then
        parity_args=(
            --global_fastpath_parity_bundle "$parity_bundle"
            --expected_global_fastpath_parity_sha256 "$parity_sha256"
        )
    else
        smplx_args=(
            --expected_smplx_asset_sha256 "$formal_smplx_sha256"
        )
    fi
    if [[ "$stage" == lower ]]; then
        lower_cache_args=(
            --use_lower_target_joints_cache true
            --lower_target_joints_cache "$lower_target_cache"
            --lower_target_joints_cache_manifest "$lower_target_manifest"
            --expected_lower_target_joints_cache_manifest_sha256 \
                "$lower_target_manifest_sha256"
            --lower_target_joints_cache_checker_receipt \
                "$lower_target_checker"
            --expected_lower_target_joints_cache_checker_sha256 \
                "$lower_target_checker_sha256"
        )
    fi

    launch_registration_in_progress=true
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT="$port"
        export PYTHONHASHSEED=2021
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        if [[ "$stage" == global ]]; then
            export SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH="$global_foot_fastpath"
        else
            unset SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH || true
        fi
        cd "$repo_root"
        exec "$python_bin" -m torch.distributed.run \
            --nproc_per_node=1 \
            --master_addr=127.0.0.1 \
            --master_port="$port" \
            show_base_train.py \
            --config "$config" \
            --formal_stage "$stage" \
            --train_only true \
            --train_rvq \
            --dataset show_base \
            --training_speakers 0 1 2 3 \
            --train_path "$rep_lmdb" \
            --data_path_1 "$asset_root/" \
            --out_path "$stage_out" \
            --run_name "$stage_run" \
            --notes "" \
            --final_ckpt_name "$final_name" \
            --epochs "$epochs" \
            --lineage_manifest "$lineage" \
            --dataset_summary "$rep_summary" \
            --expected_train_samples "$train_samples" \
            --expected_updates_per_epoch "$updates_per_epoch" \
            --strict_finite true \
            --save_every 5 \
            --log_period 1989 \
            --loader_workers "${SEMTALK_LOADER_WORKERS:-4}" \
            --random_seed 2021 \
            --pretrain false \
            --sparse 0 \
            --word_cache false \
            --word_rep disabled_zero_placeholder \
            --t_pre_encoder disabled \
            --word_index_num 0 \
            --word_dims 0 \
            --word_f 0 \
            --freeze_wordembed true \
            --hubert_mean_path "" \
            --hubert_std_path "" \
            --audio_infer_path "" \
            --base_ckpt "" \
            --test_ckpt "" \
            --load_ckpt "" \
            "${smplx_args[@]}" \
            "${parity_args[@]}" \
            "${lower_cache_args[@]}" \
            "${resume_args[@]}"
    ) >"$log_path" 2>&1 &
    local child_pid=$!
    pending_pid=$child_pid
    launch_registration_in_progress=false
    if ((pending_signal_rc != 0)); then
        local deferred_signal_rc=$pending_signal_rc
        pending_signal_rc=0
        on_signal "$deferred_signal_rc"
    fi
    local identity
    if ! identity=$(capture_child_identity "$child_pid" "$stage_run"); then
        cleanup_pending_child || true
        return 1
    fi
    local child_start=${identity%% *}
    local child_cmd_sha=${identity##* }
    child_pids+=("$child_pid")
    child_name_by_pid["$child_pid"]="$stage"
    child_start_by_pid["$child_pid"]="$child_start"
    child_cmd_sha_by_pid["$child_pid"]="$child_cmd_sha"
    active_pids+=("$child_pid")
    pending_pid=
}

launch_stage face 0 29611 configs/cnn_vqvae_face_30.yaml rvq_face_600.bin 600
launch_stage hands 1 29612 configs/cnn_vqvae_hands_30.yaml rvq_hands_500.bin 500
launch_stage upper 2 29613 configs/cnn_vqvae_upper_30.yaml rvq_upper_500.bin 500
launch_stage lower 3 29614 configs/cnn_vqvae_lower_30.yaml rvq_lower_600.bin 600
launch_stage global 4 29615 configs/cnn_vqvae_lower_foot_30.yaml last_1700_foot.bin 1700

overall_rc=0
while ((${#active_pids[@]} > 0)); do
    finished_pid=
    set +e
    wait -n -p finished_pid "${active_pids[@]}"
    child_rc=$?
    set -e
    if [[ -z "$finished_pid" ]]; then
        printf 'wait -n returned without an exact child PID\n' >&2
        overall_rc=1
        terminate_children
        break
    fi
    name=${child_name_by_pid[$finished_pid]:-unknown}
    remaining=()
    for pid in "${active_pids[@]}"; do
        if [[ "$pid" != "$finished_pid" ]]; then
            remaining+=("$pid")
        fi
    done
    active_pids=("${remaining[@]}")
    if ((child_rc != 0)); then
        printf 'stage failed: %s pid=%s rc=%s\n' \
            "$name" "$finished_pid" "$child_rc" >&2
        overall_rc=1
        terminate_children
        break
    fi
    printf 'stage complete: %s pid=%s\n' "$name" "$finished_pid"
done
trap - EXIT INT TERM
exit "$overall_rc"
