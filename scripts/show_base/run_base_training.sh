#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Must run beneath /tmp/globaldiff_guarded_runner.py.  Base remains one-GPU to
# preserve the released global batch and BatchNorm/update semantics.

if [[ $# -ne 9 && $# -ne 10 ]]; then
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON BASE_LMDB BASE_SUMMARY LINEAGE ASSET_ROOT OUTPUT_ROOT RUN_ID GPU [--resume]"
    exit 2
fi

repo_root=$1
python_bin=$2
base_lmdb=$3
base_summary=$4
lineage=$5
asset_root=$6
output_root=$7
run_id=$8
gpu=$9
resume_mode=false
if [[ $# -eq 10 ]]; then
    if [[ ${10} != "--resume" ]]; then
        printf 'the only supported tenth argument is --resume\n' >&2
        exit 2
    fi
    resume_mode=true
fi

for required in "$repo_root/show_base_train.py" "$python_bin" "$base_summary" "$lineage"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ ! -d "$base_lmdb" ]]; then
    printf 'missing Base LMDB: %s\n' "$base_lmdb" >&2
    exit 1
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'unsafe run id: %s\n' "$run_id" >&2
    exit 1
fi

read -r train_samples updates_per_epoch < <(
    "$python_bin" - "$base_summary" "$base_lmdb" "$lineage" <<'PY'
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
summary = json.loads(summary_path.read_text())
if (
    summary.get("status") != "complete"
    or summary.get("format") != "semtalk_show_base_lmdb_summary_v1"
):
    raise SystemExit("Base summary is not complete")
lineage_digest_state = hashlib.sha256()
with lineage_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        lineage_digest_state.update(chunk)
if (
    Path(summary.get("lineage_json", "")).resolve() != lineage_path
    or summary.get("lineage_json_sha256")
    != lineage_digest_state.hexdigest()
):
    raise SystemExit("Base summary/lineage binding mismatch")
if Path(summary["lmdb"]).resolve() != lmdb_path:
    raise SystemExit("Base LMDB path mismatch")
digest_state = hashlib.sha256()
with (lmdb_path / "data.mdb").open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest_state.update(chunk)
digest = digest_state.hexdigest()
if digest != summary["data_mdb_sha256"]:
    raise SystemExit("Base data.mdb SHA mismatch")
entries = require_exact_int(summary.get("entries"), "Base entries")
updates = entries // 64
if entries != 127_286 or updates != 1_988:
    raise SystemExit(f"formal Base accounting mismatch: {entries=} {updates=}")
print(entries, updates)
PY
)

stage_out="$output_root/base/"
stage_run="${run_id}_base"
stage_dir="$stage_out/custom/$stage_run"
resume_args=()
log_path="$output_root/logs/$run_id/base.log"
if [[ "$resume_mode" == false && -e "$stage_dir" ]]; then
    printf 'refusing to reuse Base output: %s\n' "$stage_out/custom/$stage_run" >&2
    exit 1
fi
if [[ "$resume_mode" == true ]]; then
    if [[ ! -d "$stage_dir" ]]; then
        printf 'resume Base output is missing: %s\n' "$stage_dir" >&2
        exit 1
    fi
    if [[ ! -f "$stage_dir/latest_resume.pt" ]]; then
        printf 'resume checkpoint is missing: %s\n' \
            "$stage_dir/latest_resume.pt" >&2
        exit 1
    fi
    resume_args=(--resume_state "$stage_dir/latest_resume.pt")
    log_path="$output_root/logs/$run_id/base.resume.$$.log"
fi
mkdir -p "$stage_out" "$output_root/logs/$run_id"

export CUDA_VISIBLE_DEVICES="$gpu"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29621
export PYTHONHASHSEED=43
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$repo_root"
exec "$python_bin" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29621 \
    show_base_train.py \
    --config configs/semtalk_base.yaml \
    --formal_stage base \
    --train_only true \
    --dataset show_base \
    --training_speakers 0 1 2 3 \
    --train_path "$base_lmdb" \
    --data_path_1 "$asset_root/" \
    --out_path "$stage_out" \
    --run_name "$stage_run" \
    --notes "" \
    --final_ckpt_name semtalk_base_epoch_400.bin \
    --epochs 400 \
    --lineage_manifest "$lineage" \
    --dataset_summary "$base_summary" \
    --expected_train_samples "$train_samples" \
    --expected_updates_per_epoch "$updates_per_epoch" \
    --strict_finite true \
    --save_every 5 \
    --log_period "$updates_per_epoch" \
    --loader_workers "${SEMTALK_LOADER_WORKERS:-4}" \
    --random_seed 43 \
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
    "${resume_args[@]}" \
    >"$log_path" 2>&1
