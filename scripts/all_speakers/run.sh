#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

SPEAKERS=(1 2 3 4 5 6 7 9 10 11 12 13 15 16 17 18 20 21 22 23 24 25 27 28 30)
DATA_ROOT=${SEMTALK_DATA_ROOT:-./BEAT2/beat_english_v2.0.0}
DATASET_ROOT=${SEMTALK_ALL_SPEAKER_DATASET_ROOT:-./datasets/all_speakers}
WEIGHT_ROOT=${SEMTALK_ALL_SPEAKER_WEIGHT_ROOT:-./weights/all_speakers}
OUTPUT_ROOT=${SEMTALK_ALL_SPEAKER_OUTPUT_ROOT:-./outputs/all_speakers}
PYTHON=${PYTHON:-python}
SEED=${SEMTALK_SEED:-43}
VQ_SEED=${SEMTALK_VQ_SEED:-2021}

export SEMTALK_PRETRAINED_VQ_DIR=${SEMTALK_PRETRAINED_VQ_DIR:-$WEIGHT_ROOT/pretrained_vq}

usage() {
    printf '%s\n' \
        "Usage: scripts/all_speakers/run.sh COMMAND" \
        "" \
        "Commands:" \
        "  prepare-train   Build the 25-speaker training LMDB" \
        "  prepare-test    Build the 25-speaker test pickle" \
        "  vq-face         Train the face RVQ-VAE" \
        "  vq-hands        Train the hands RVQ-VAE" \
        "  vq-upper        Train the upper-body RVQ-VAE" \
        "  vq-lower        Train the lower-body RVQ-VAE" \
        "  vae-global      Train the global/root-motion VAE" \
        "  base            Train Base Motion Generation" \
        "  sparse          Train Sparse Motion Generation" \
        "  test            Evaluate the released all-speaker Sparse checkpoint"
}

run_vq() {
    local config=$1
    local name=$2
    local epochs=$3
    "$PYTHON" train.py \
        --config "$config" \
        --train_rvq \
        --training_speakers "${SPEAKERS[@]}" \
        --data_path "$DATA_ROOT" \
        --cache_path "$DATASET_ROOT/cache/$name" \
        --out_path "$OUTPUT_ROOT/$name/" \
        --notes "_all_speakers" \
        --random_seed "$VQ_SEED" \
        --epochs "$epochs"
}

command=${1:-}
case "$command" in
    prepare-train)
        mkdir -p "$DATASET_ROOT"
        "$PYTHON" dataloaders/save_train_dataset.py \
            --training_speakers "${SPEAKERS[@]}" \
            --data_path "$DATA_ROOT/" \
            --cache_path "$DATASET_ROOT/cache" \
            --dst_lmdb "$DATASET_ROOT/beat2_semtalk_train"
        ;;
    prepare-test)
        mkdir -p "$DATASET_ROOT"
        "$PYTHON" dataloaders/save_test_dataset.py \
            --training_speakers "${SPEAKERS[@]}" \
            --data_path "$DATA_ROOT/" \
            --cache_path "$DATASET_ROOT/cache" \
            --dst_pkl "$DATASET_ROOT/beat2_semtalk_test.pkl"
        ;;
    vq-face)
        run_vq configs/cnn_vqvae_face_30.yaml vq-face 600
        ;;
    vq-hands)
        run_vq configs/cnn_vqvae_hands_30.yaml vq-hands 500
        ;;
    vq-upper)
        run_vq configs/cnn_vqvae_upper_30.yaml vq-upper 500
        ;;
    vq-lower)
        run_vq configs/cnn_vqvae_lower_30.yaml vq-lower 600
        ;;
    vae-global)
        run_vq configs/cnn_vqvae_lower_foot_30.yaml vae-global 1700
        ;;
    base)
        "$PYTHON" train.py \
            --config configs/semtalk_base.yaml \
            --training_speakers "${SPEAKERS[@]}" \
            --data_path "$DATA_ROOT/" \
            --train_path "$DATASET_ROOT/beat2_semtalk_train" \
            --test_path "$DATASET_ROOT/beat2_semtalk_test.pkl" \
            --out_path "$OUTPUT_ROOT/base/seed-$SEED/" \
            --notes "_all_speakers_seed_$SEED" \
            --random_seed "$SEED" \
            --epochs 400
        ;;
    sparse)
        "$PYTHON" train.py \
            --config configs/semtalk_sparse.yaml \
            --training_speakers "${SPEAKERS[@]}" \
            --data_path "$DATA_ROOT/" \
            --train_path "$DATASET_ROOT/beat2_semtalk_train" \
            --test_path "$DATASET_ROOT/beat2_semtalk_test.pkl" \
            --base_ckpt "$WEIGHT_ROOT/best_semtalk_base.bin" \
            --out_path "$OUTPUT_ROOT/sparse/seed-$SEED/" \
            --notes "_all_speakers_seed_$SEED" \
            --random_seed "$SEED" \
            --epochs 400
        ;;
    test)
        "$PYTHON" train.py \
            --config configs/semtalk_sparse.yaml \
            --training_speakers "${SPEAKERS[@]}" \
            --data_path "$DATA_ROOT/" \
            --train_path "$DATASET_ROOT/beat2_semtalk_train" \
            --test_path "$DATASET_ROOT/beat2_semtalk_test.pkl" \
            --base_ckpt "$WEIGHT_ROOT/best_semtalk_base.bin" \
            --load_ckpt "$WEIGHT_ROOT/best_semtalk_sparse.bin" \
            --out_path "$OUTPUT_ROOT/test/" \
            --notes "_all_speakers" \
            --test_state
        ;;
    *)
        usage
        exit 2
        ;;
esac
