#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Formal launcher for the exact official BEAT2 All-Speakers Base + five
# representation files.  Canonical/audio/source/output arguments remain caller
# supplied; SHOW training artifacts are forbidden.  GPU execution must stay below
# /tmp/globaldiff_guarded_runner.py.

if [[ $# -lt 10 ]]; then
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON BASE FACE UPPER HANDS LOWER GLOBAL CROSS_DOMAIN_GATE CROSS_DOMAIN_GATE_SHA256 [INFERENCE_ARGS...]"
    exit 2
fi

repo_root=$1
python_bin=$2
base_checkpoint=$3
face_checkpoint=$4
upper_checkpoint=$5
hands_checkpoint=$6
lower_checkpoint=$7
global_checkpoint=$8
cross_domain_gate=$9
cross_domain_gate_sha256=${10}
shift 10

declare -A expected_names=(
    [base]=best_semtalk_base.bin
    [face]=rvq_face_600.bin
    [upper]=rvq_upper_500.bin
    [hands]=rvq_hands_500.bin
    [lower]=rvq_lower_600.bin
    [global]=last_1700_foot.bin
)
declare -A expected_hashes=(
    [base]=52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603
    [face]=31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110a6152127
    [upper]=05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d88d08ac17
    [hands]=08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a539b03e436
    [lower]=2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c171ae4efce8
    [global]=6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e0144ee12
)
declare -A checkpoint_paths=(
    [base]="$base_checkpoint"
    [face]="$face_checkpoint"
    [upper]="$upper_checkpoint"
    [hands]="$hands_checkpoint"
    [lower]="$lower_checkpoint"
    [global]="$global_checkpoint"
)

for required in \
    "$repo_root/scripts/show_base/run_base_inference.py" \
    "$python_bin"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        printf 'missing/unsafe required input: %s\n' "$required" >&2
        exit 1
    fi
done
for stage in base face upper hands lower global; do
    checkpoint=${checkpoint_paths[$stage]}
    if [[ ! -f "$checkpoint" || -L "$checkpoint" ]]; then
        printf 'missing/unsafe %s checkpoint: %s\n' "$stage" "$checkpoint" >&2
        exit 1
    fi
    if [[ ${checkpoint##*/} != "${expected_names[$stage]}" ]]; then
        printf 'invalid %s checkpoint basename: %s\n' "$stage" "$checkpoint" >&2
        exit 2
    fi
done
if [[ ! -f "$cross_domain_gate" || -L "$cross_domain_gate" ]]; then
    printf 'missing/unsafe released cross-domain gate: %s\n' \
        "$cross_domain_gate" >&2
    exit 1
fi
if [[ ! "$cross_domain_gate_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'invalid released cross-domain gate SHA-256: %s\n' \
        "$cross_domain_gate_sha256" >&2
    exit 2
fi

for argument in "$@"; do
    case "$argument" in
        --prerequisite-source|--prerequisite-source=*|\
        --base-checkpoint-source|--base-checkpoint-source=*|\
        --released-cross-domain-gate-json|\
        --released-cross-domain-gate-json=*|\
        --expected-released-cross-domain-gate-sha256|\
        --expected-released-cross-domain-gate-sha256=*|\
        --base-training-lineage-manifest|\
        --base-training-lineage-manifest=*|\
        --base-training-summary-json|--base-training-summary-json=*|\
        --representation-training-lineage-manifest|\
        --representation-training-lineage-manifest=*|\
        --expected-training-source-commit|\
        --expected-training-source-commit=*|\
        --expected-training-source-tree|\
        --expected-training-source-tree=*|\
        --base-checkpoint|--base-checkpoint=*|\
        --expected-base-sha256|--expected-base-sha256=*|\
        --base-status-json|--base-status-json=*|\
        --base-candidate-manifest|--base-candidate-manifest=*|\
        --expected-base-candidate-manifest-sha256|\
        --expected-base-candidate-manifest-sha256=*|\
        --expected-base-formal-status-sha256|\
        --expected-base-formal-status-sha256=*|\
        --expected-base-final-checkpoint-sha256|\
        --expected-base-final-checkpoint-sha256=*|\
        --face-checkpoint|--face-checkpoint=*|\
        --expected-face-sha256|--expected-face-sha256=*|\
        --face-status-json|--face-status-json=*|\
        --upper-checkpoint|--upper-checkpoint=*|\
        --expected-upper-sha256|--expected-upper-sha256=*|\
        --upper-status-json|--upper-status-json=*|\
        --hands-checkpoint|--hands-checkpoint=*|\
        --expected-hands-sha256|--expected-hands-sha256=*|\
        --hands-status-json|--hands-status-json=*|\
        --lower-checkpoint|--lower-checkpoint=*|\
        --expected-lower-sha256|--expected-lower-sha256=*|\
        --lower-status-json|--lower-status-json=*|\
        --global-checkpoint|--global-checkpoint=*|\
        --expected-global-sha256|--expected-global-sha256=*|\
        --global-status-json|--global-status-json=*)
            printf 'official release trust-root argument is launcher-owned: %s\n' \
                "$argument" >&2
            exit 2
            ;;
    esac
done

exec "$python_bin" "$repo_root/scripts/show_base/run_base_inference.py" \
    --prerequisite-source released_all_speakers_v1 \
    --base-checkpoint-source released_all_speakers_v1 \
    --released-cross-domain-gate-json "$cross_domain_gate" \
    --expected-released-cross-domain-gate-sha256 \
    "$cross_domain_gate_sha256" \
    --base-checkpoint "$base_checkpoint" \
    --expected-base-sha256 "${expected_hashes[base]}" \
    --face-checkpoint "$face_checkpoint" \
    --expected-face-sha256 "${expected_hashes[face]}" \
    --upper-checkpoint "$upper_checkpoint" \
    --expected-upper-sha256 "${expected_hashes[upper]}" \
    --hands-checkpoint "$hands_checkpoint" \
    --expected-hands-sha256 "${expected_hashes[hands]}" \
    --lower-checkpoint "$lower_checkpoint" \
    --expected-lower-sha256 "${expected_hashes[lower]}" \
    --global-checkpoint "$global_checkpoint" \
    --expected-global-sha256 "${expected_hashes[global]}" \
    "$@"
