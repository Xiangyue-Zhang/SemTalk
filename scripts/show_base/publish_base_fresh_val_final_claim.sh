#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Stage C: CPU-only publication.  It consumes the immutable Stage-A winner and
# the separately guarded Stage-B winner-full closure, then authorizes one
# still-absent test output root.  It never runs a GPU process.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON RUN_SPEC RUN_SPEC_SHA256 RUN_SPEC_PAYLOAD_SHA256 WINNER WINNER_SHA256 WINNER_PAYLOAD_SHA256 WINNER_FULL_CLOSURE CLOSURE_SHA256 CLOSURE_PAYLOAD_SHA256 EXPECTED_TEST_OUTPUT_ROOT OUTPUT_ROOT SOURCE_COMMIT SOURCE_TREE"
}

if [[ $# -ne 15 ]]; then
    usage >&2
    exit 2
fi

repo_root=$1
python_bin=$2
run_spec=$3
run_spec_sha256=$4
run_spec_payload_sha256=$5
winner=$6
winner_sha256=$7
winner_payload_sha256=$8
closure=$9
closure_sha256=${10}
closure_payload_sha256=${11}
expected_test_output_root=${12}
output_root=${13}
source_commit=${14}
source_tree=${15}

if [[ ! "$repo_root" = /* || ! "$python_bin" = /* || \
      ! "$run_spec" = /* || ! "$winner" = /* || ! "$closure" = /* || \
      ! "$expected_test_output_root" = /* || ! "$output_root" = /* || \
      ! -x "$python_bin" || ! -f "$run_spec" || -L "$run_spec" || \
      ! -f "$winner" || -L "$winner" || \
      ! -f "$closure" || -L "$closure" || \
      -e "$expected_test_output_root" || \
      -L "$expected_test_output_root" || \
      -e "$output_root" || -L "$output_root" || \
      ! -d "$(dirname "$output_root")" ]]; then
    printf 'final claim inputs must be canonical create-new paths\n' >&2
    exit 2
fi
for digest in \
    "$run_spec_sha256" "$run_spec_payload_sha256" \
    "$winner_sha256" "$winner_payload_sha256" \
    "$closure_sha256" "$closure_payload_sha256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
for oid in "$source_commit" "$source_tree"; do
    [[ "$oid" =~ ^[0-9a-f]{40}$ ]] || exit 2
done

if [[ "$(git -C "$repo_root" remote get-url origin)" != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      "$(git -C "$repo_root" rev-parse HEAD)" != "$source_commit" || \
      "$(git -C "$repo_root" rev-parse 'HEAD^{tree}')" != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'final claim source must be exact clean detached SemTalk with zero branches\n' >&2
    exit 1
fi

orchestrator="$repo_root/scripts/show_base/base_fresh_val_orchestrator.py"
if [[ ! -f "$orchestrator" || -L "$orchestrator" ]]; then
    printf 'final claim orchestrator is unavailable\n' >&2
    exit 1
fi

mapfile -d '' -t output_root_identity < <(
    "$python_bin" "$orchestrator" create-run-root --path "$output_root" \
        --subdirectory logs
)
if [[ ${#output_root_identity[@]} -ne 3 || \
      "${output_root_identity[0]}" != "$output_root" || \
      ! ${output_root_identity[1]} =~ ^[0-9]+$ || \
      ! ${output_root_identity[2]} =~ ^[0-9]+$ ]]; then
    printf 'claim output root could not be created safely\n' >&2
    exit 1
fi
"$python_bin" "$orchestrator" validate-run-spec \
    --run-spec-path "$run_spec" \
    --run-spec-sha256 "$run_spec_sha256" \
    --run-spec-payload-sha256 "$run_spec_payload_sha256" \
    --source-commit "$source_commit" --source-tree "$source_tree" \
    --output-json "$output_root/run-spec-audit.json" \
    >"$output_root/logs/run-spec.log" 2>&1

mapfile -d '' -t common < <(
    "$python_bin" "$orchestrator" extract-run-spec \
        --run-spec-path "$run_spec" \
        --run-spec-sha256 "$run_spec_sha256" \
        --run-spec-payload-sha256 "$run_spec_payload_sha256" \
        --source-commit "$source_commit" --source-tree "$source_tree" \
        --profile finalizer
)
if [[ ${#common[@]} -lt 12 || $(((${#common[@]} - 12) % 4)) -ne 0 ]]; then
    printf 'run-spec final publication extraction failed\n' >&2
    exit 1
fi
prerequisite=${common[0]}
prerequisite_sha=${common[1]}
prerequisite_payload=${common[3]}
continuation=${common[4]}
continuation_sha=${common[5]}
continuation_payload=${common[7]}
real_cache=${common[8]}
real_cache_sha=${common[9]}
real_cache_payload=${common[11]}
continuation_wave_args=()
for ((index = 12; index < ${#common[@]}; index += 4)); do
    continuation_wave_args+=(
        --continuation-wave-path "${common[$index]}"
        --continuation-wave-sha256 "${common[$((index + 1))]}"
        --continuation-wave-bytes "${common[$((index + 2))]}"
        --continuation-wave-payload-sha256 "${common[$((index + 3))]}"
    )
done

claim="$output_root/base-published-test-winner-claim.json"
"$python_bin" "$orchestrator" publish-fresh-claim \
    --winner-selection-path "$winner" \
    --winner-selection-sha256 "$winner_sha256" \
    --winner-selection-payload-sha256 "$winner_payload_sha256" \
    --prerequisite-path "$prerequisite" \
    --prerequisite-sha256 "$prerequisite_sha" \
    --prerequisite-payload-sha256 "$prerequisite_payload" \
    --continuation-path "$continuation" \
    --continuation-sha256 "$continuation_sha" \
    --continuation-payload-sha256 "$continuation_payload" \
    --real-feature-cache-path "$real_cache" \
    --real-feature-cache-sha256 "$real_cache_sha" \
    --real-feature-cache-payload-sha256 "$real_cache_payload" \
    --winner-full-closure-path "$closure" \
    --winner-full-closure-sha256 "$closure_sha256" \
    --winner-full-closure-payload-sha256 "$closure_payload_sha256" \
    "${continuation_wave_args[@]}" \
    --expected-test-output-root "$expected_test_output_root" \
    --output-json "$claim" >"$output_root/logs/claim.log" 2>&1

mapfile -d '' -t claim_fields < <(
    "$python_bin" "$orchestrator" artifact-fields \
        --artifact-path "$claim" --payload-receipt
)
if [[ ${#claim_fields[@]} -ne 3 ]]; then
    printf 'published claim could not be pinned\n' >&2
    exit 1
fi
printf '%s\0%s\0%s\0' \
    "$claim" "${claim_fields[0]}" "${claim_fields[2]}"
