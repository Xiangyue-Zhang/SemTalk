#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# This launcher must itself run as the child of /tmp/globaldiff_guarded_runner.py
# with physical GPUs 0..7 reserved.  The required fixed partition is selected
# by SEMTALK_FORMAL_PARTITION:
#
#   master: Face=0..3; Hands=4..7; Global starts on the first released group
#   worker: Upper=0..3; Lower=4..7
#
# Every RVQ is one single-node W4 DDP job with local batch 64 and locked global
# batch 256, matching the proven All-Speakers prerequisite exposure topology.
# Global retains its exact W1 fastpath with batch 256 and overlaps the slower
# master RVQ after the first four-GPU group is released. No NCCL process group
# crosses the master/worker node boundary.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON REP_LMDB REP_SUMMARY LINEAGE ASSET_ROOT OUTPUT_ROOT RUN_ID PARITY_BUNDLE PARITY_SHA256 [--resume | --continuation-wave WAVE_JSON WAVE_SHA256]"
}

if [[ $# -ne 10 && $# -ne 11 && $# -ne 13 ]]; then
    usage
    exit 2
fi

launcher_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=guarded_runner_contract.sh
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

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
resume_mode=false
continuation_mode=false
continuation_wave=
continuation_wave_sha256=
formal_smplx_sha256=bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74
official_vq_root=${SEMTALK_OFFICIAL_VQ_ROOT:-/local-ssd/xiangyuezhang/semtalk_all_speakers_full_vq_20260730/pretrained_vq}
if [[ $# -eq 11 ]]; then
    if [[ ${11} != "--resume" ]]; then
        usage
        exit 2
    fi
    resume_mode=true
elif [[ $# -eq 13 ]]; then
    if [[ ${11} != "--continuation-wave" ]]; then
        usage
        exit 2
    fi
    continuation_mode=true
    continuation_wave=${12}
    continuation_wave_sha256=${13}
fi

formal_partition=${SEMTALK_FORMAL_PARTITION:-}
case "$formal_partition" in
    master)
        active_stages=(face hands global)
        ;;
    worker)
        active_stages=(upper lower)
        ;;
    *)
        printf '%s\n' \
            "SEMTALK_FORMAL_PARTITION must be exactly master or worker" >&2
        exit 2
        ;;
esac

for required in "$repo_root/show_base_train.py" "$python_bin" "$rep_summary" \
    "$lineage" "$parity_bundle"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ "$continuation_mode" == true ]]; then
    if [[ ! -f "$continuation_wave" || -L "$continuation_wave" ]]; then
        printf 'missing regular continuation wave: %s\n' \
            "$continuation_wave" >&2
        exit 1
    fi
    if [[ ! "$continuation_wave_sha256" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'invalid continuation wave SHA-256\n' >&2
        exit 1
    fi
fi
if [[ ! -d "$rep_lmdb" ]]; then
    printf 'missing representation LMDB: %s\n' "$rep_lmdb" >&2
    exit 1
fi
declare -A official_filename=(
    [face]=rvq_face_600.bin
    [hands]=rvq_hands_500.bin
    [upper]=rvq_upper_500.bin
    [lower]=rvq_lower_600.bin
    [global]=last_1700_foot.bin
)
declare -A official_sha256=(
    [face]=31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110a6152127
    [hands]=08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a539b03e436
    [upper]=05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d88d08ac17
    [lower]=2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c171ae4efce8
    [global]=6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e0144ee12
)
for stage in "${active_stages[@]}"; do
    checkpoint="$official_vq_root/${official_filename[$stage]}"
    if [[ ! -f "$checkpoint" || -L "$checkpoint" ]]; then
        printf 'missing regular official %s checkpoint: %s\n' \
            "$stage" "$checkpoint" >&2
        exit 1
    fi
    if [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" != \
          "${official_sha256[$stage]}" ]]; then
        printf 'official %s checkpoint SHA-256 mismatch\n' "$stage" >&2
        exit 1
    fi
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'unsafe run id: %s\n' "$run_id" >&2
    exit 1
fi

read -r train_samples updates_per_epoch global_foot_fastpath < <(
    "$python_bin" - "$rep_summary" "$rep_lmdb" "$lineage" \
        "$parity_bundle" "$parity_sha256" "$asset_root" <<'PY'
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
entries = require_exact_int(summary.get("entries"), "representation entries")
updates = entries // 256
if entries != 127_286 or updates != 497:
    raise SystemExit(
        f"formal representation accounting mismatch: {entries=} {updates=}"
    )
print(entries, updates, fastpath)
PY
)

declare -A continuation_resume_path=()
declare -A continuation_new_run_path=()
continuation_boundary_epoch=0
continuation_target_epoch=200
if [[ "$continuation_mode" == true ]]; then
    mapfile -d '' -t continuation_fields < <(
        "$python_bin" - "$repo_root" "$continuation_wave" \
            "$continuation_wave_sha256" "$formal_partition" \
            "$output_root" "$run_id" <<'PY'
from pathlib import Path
import socket
import sys

repo = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(repo))
from scripts.show_base import prerequisite_continuation_wave as wave

wave_path = Path(sys.argv[2])
wave_sha = sys.argv[3]
partition = sys.argv[4]
output_root = Path(sys.argv[5])
run_id = sys.argv[6]
receipt = wave.replay_wave_file(wave_path, wave_sha)
expected = {
    "master": ("face", "hands", "global"),
    "worker": ("upper", "lower"),
}[partition]
entries = {entry["stage"]: entry for entry in receipt["stages"]}
if set(entries) != set(wave.STAGES):
    raise SystemExit("continuation wave does not cover exact five stages")
hostname = socket.gethostname()
fields = [
    str(receipt["boundary_epoch"]),
    str(receipt["target_epoch"]),
]
for stage in expected:
    entry = entries[stage]
    old = entry["old_segment"]
    new = entry["new_segment"]
    expected_run = output_root / stage / "custom" / f"{run_id}_{stage}"
    if (
        new["host"] != hostname
        or new["run_path"] != str(expected_run)
        or old["boundary_resume"]["path"]
        != str(Path(old["run_path"]) / "latest_resume.pt")
    ):
        raise SystemExit(
            f"{stage} continuation host/run/resume binding mismatch"
        )
    fields.extend(
        (
            stage,
            old["boundary_resume"]["path"],
            new["run_path"],
        )
    )
sys.stdout.buffer.write(b"\0".join(item.encode() for item in fields) + b"\0")
PY
    )
    expected_field_count=$((2 + 3 * ${#active_stages[@]}))
    if [[ ${#continuation_fields[@]} -ne $expected_field_count ]]; then
        printf 'continuation wave preflight returned incomplete fields\n' >&2
        exit 1
    fi
    continuation_boundary_epoch=${continuation_fields[0]}
    continuation_target_epoch=${continuation_fields[1]}
    if ((continuation_target_epoch != continuation_boundary_epoch + 20)); then
        printf 'continuation wave is not exact +20\n' >&2
        exit 1
    fi
    field_index=2
    while ((field_index < ${#continuation_fields[@]})); do
        stage=${continuation_fields[$field_index]}
        continuation_resume_path[$stage]=${continuation_fields[$((field_index + 1))]}
        continuation_new_run_path[$stage]=${continuation_fields[$((field_index + 2))]}
        field_index=$((field_index + 3))
    done
fi

mkdir -p "$output_root/logs/$run_id"
for stage in "${active_stages[@]}"; do
    stage_dir="$output_root/$stage/custom/${run_id}_${stage}"
    if [[ "$continuation_mode" == true ]]; then
        if [[ "$stage_dir" != "${continuation_new_run_path[$stage]:-}" ]]; then
            printf 'continuation new segment path mismatch: %s\n' \
                "$stage" >&2
            exit 1
        fi
        if [[ -e "$stage_dir" || -L "$stage_dir" ]]; then
            printf 'refusing to reuse continuation segment: %s\n' \
                "$stage_dir" >&2
            exit 1
        fi
    elif [[ "$resume_mode" == false && -e "$stage_dir" ]]; then
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
    local physical_gpus=$2
    local port=$3
    local config=$4
    local final_name=$5
    local epochs=$6
    local pool_mode=${7:-disabled}
    local helper_devices=${8:-}
    local gate_report=${9:-}
    local gate_sha256=${10:-}
    local stage_out="$output_root/$stage/"
    local stage_run="${run_id}_${stage}"
    local stage_dir="$stage_out/custom/$stage_run"
    local resume_args=()
    local parity_args=()
    local smplx_args=()
    local lower_backend_args=()
    local -a stage_gpus=()
    local log_path="$output_root/logs/$run_id/$stage.log"
    mkdir -p "$stage_out"

    if [[ "$continuation_mode" == true ]]; then
        epochs=$continuation_target_epoch
        final_name="show_ft_${stage}_${continuation_target_epoch}.bin"
    fi

    IFS=, read -r -a stage_gpus <<<"$physical_gpus"
    if [[ "$pool_mode" == disabled ]]; then
        local expected_world_size=4
        if [[ "$stage" == global ]]; then
            expected_world_size=1
        fi
        if [[ ${#stage_gpus[@]} -ne $expected_world_size ]]; then
            printf '%s requires exactly %s physical GPUs\n' \
                "$stage" "$expected_world_size" >&2
            return 1
        fi
        if [[ -n "$helper_devices" || -n "$gate_report" || \
              -n "$gate_sha256" ]]; then
            printf '%s stock path forbids helper/gate inputs\n' "$stage" >&2
            return 1
        fi
    else
        printf 'unsupported stage pool mode: %s\n' "$pool_mode" >&2
        return 1
    fi

    if [[ "$continuation_mode" == true ]]; then
        if [[ ! -f "${continuation_resume_path[$stage]:-}" || \
              -L "${continuation_resume_path[$stage]:-}" ]]; then
            printf 'continuation boundary resume is missing: %s\n' \
                "$stage" >&2
            return 1
        fi
        resume_args=(
            --resume_state "${continuation_resume_path[$stage]}"
            --resume_wave_receipt "$continuation_wave"
            --expected_resume_wave_sha256 "$continuation_wave_sha256"
        )
        log_path="$output_root/logs/$run_id/$stage.continuation.e${continuation_target_epoch}.$$.log"
    elif [[ "$resume_mode" == true ]]; then
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
            --smplx_training_pool_mode disabled
        )
    fi
    if [[ "$stage" == lower ]]; then
        lower_backend_args=(
            --use_lower_target_joints_cache false
        )
    fi

    launch_registration_in_progress=true
    (
        export CUDA_VISIBLE_DEVICES="$physical_gpus"
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
            --nproc_per_node="${#stage_gpus[@]}" \
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
            --batch_size "$((256 / ${#stage_gpus[@]}))" \
            --global_batch_size 256 \
            --initial-model-checkpoint \
                "$official_vq_root/${official_filename[$stage]}" \
            --strict_finite true \
            --rvq_check_finite_every_step false \
            --save_every 5 \
            --log_period "$updates_per_epoch" \
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
            "${lower_backend_args[@]}" \
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

if [[ "$formal_partition" == master ]]; then
    launch_stage \
        face 0,1,2,3 29611 configs/cnn_vqvae_face_30.yaml \
        show_ft_face_200.bin 200 disabled
    launch_stage \
        hands 4,5,6,7 29612 configs/cnn_vqvae_hands_30.yaml \
        show_ft_hands_200.bin 200 disabled
else
    launch_stage \
        upper 0,1,2,3 29613 configs/cnn_vqvae_upper_30.yaml \
        show_ft_upper_200.bin 200 disabled
    launch_stage \
        lower 4,5,6,7 29614 configs/cnn_vqvae_lower_30.yaml \
        show_ft_lower_200.bin 200 disabled
fi

overall_rc=0
global_launched=false
wait_for_active() {
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
        return 1
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
    if [[ "$formal_partition" == master && \
          "$global_launched" == false && \
          ( "$name" == face || "$name" == hands ) ]]; then
        global_gpu=0
        if [[ "$name" == hands ]]; then
            global_gpu=4
        fi
        if ! launch_stage \
            global "$global_gpu" 29615 \
            configs/cnn_vqvae_lower_foot_30.yaml \
            show_ft_global_200.bin 200 disabled; then
            printf 'failed to launch global after %s completed\n' \
                "$name" >&2
            overall_rc=1
            terminate_children
            break
        fi
        global_launched=true
        printf 'global launched on released GPU %s after %s\n' \
            "$global_gpu" "$name"
    fi
done
}

wait_for_active || true
trap - EXIT INT TERM
exit "$overall_rc"
