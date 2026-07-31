#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# This launcher only fixes the Base-specific formal trust roots.  The caller
# must supply the remaining canonical/audio/prerequisite/output arguments
# accepted by run_base_inference.py.  GPU execution must remain beneath
# /tmp/globaldiff_guarded_runner.py.

if [[ $# -lt 9 ]]; then
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON BASE_CANDIDATE BASE_CANDIDATE_SHA BASE_CANDIDATE_MANIFEST MANIFEST_SHA BASE_STATUS STATUS_SHA FINAL_CHECKPOINT_SHA [INFERENCE_ARGS...]"
    exit 2
fi

launcher_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=guarded_runner_contract.sh
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

repo_root=$1
python_bin=$2
base_candidate=$3
base_candidate_sha=$4
base_candidate_manifest=$5
base_candidate_manifest_sha=$6
base_status=$7
base_status_sha=$8
base_final_checkpoint_sha=$9
shift 9

for required in \
    "$repo_root/scripts/show_base/run_base_inference.py" \
    "$python_bin" \
    "$base_candidate" \
    "$base_candidate_manifest" \
    "$base_status"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    fi
done

for digest in \
    "$base_candidate_sha" \
    "$base_candidate_manifest_sha" \
    "$base_status_sha" \
    "$base_final_checkpoint_sha"; do
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'invalid lowercase SHA-256: %s\n' "$digest" >&2
        exit 2
    fi
done

for argument in "$@"; do
    case "$argument" in
        --base-checkpoint|--base-checkpoint=*|\
        --expected-base-sha256|--expected-base-sha256=*|\
        --base-status-json|--base-status-json=*|\
        --base-candidate-manifest|--base-candidate-manifest=*|\
        --expected-base-candidate-manifest-sha256|\
        --expected-base-candidate-manifest-sha256=*|\
        --expected-base-formal-status-sha256|\
        --expected-base-formal-status-sha256=*|\
        --expected-base-final-checkpoint-sha256|\
        --expected-base-final-checkpoint-sha256=*)
            printf 'Base formal trust-root argument is launcher-owned: %s\n' \
                "$argument" >&2
            exit 2
            ;;
    esac
done

exec "$python_bin" "$repo_root/scripts/show_base/run_base_inference.py" \
    --base-checkpoint "$base_candidate" \
    --expected-base-sha256 "$base_candidate_sha" \
    --base-status-json "$base_status" \
    --base-candidate-manifest "$base_candidate_manifest" \
    --expected-base-candidate-manifest-sha256 \
        "$base_candidate_manifest_sha" \
    --expected-base-formal-status-sha256 "$base_status_sha" \
    --expected-base-final-checkpoint-sha256 \
        "$base_final_checkpoint_sha" \
    "$@"
