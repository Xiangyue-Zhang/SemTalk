#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Stage A: CPU-only immutable exact union and primary-screen winner selection.
# Positional partition triples are ordered by partition_id and each is
# PATH SHA256 RECEIPT_PAYLOAD_SHA256.  This stage never runs full metrics and
# never publishes test authority; the winner-only guarded Stage B must finish
# first.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON RUN_SPEC RUN_SPEC_SHA256 RUN_SPEC_PAYLOAD_SHA256 PARTITION_COUNT OUTPUT_ROOT SOURCE_COMMIT SOURCE_TREE [PARTITION_PATH PARTITION_SHA256 PARTITION_PAYLOAD_SHA256]..."
}

if [[ $# -lt 12 ]]; then
    usage >&2
    exit 2
fi

repo_root=$1
python_bin=$2
run_spec=$3
run_spec_sha256=$4
run_spec_payload_sha256=$5
partition_count=$6
output_root=$7
source_commit=$8
source_tree=$9
shift 9

if [[ ! "$partition_count" =~ ^[1-9][0-9]*$ || \
      $# -ne $((partition_count * 3)) ]]; then
    printf 'partition triples do not match partition_count\n' >&2
    exit 2
fi
if [[ ! "$repo_root" = /* || ! "$python_bin" = /* || \
      ! "$run_spec" = /* || ! "$output_root" = /* || \
      -e "$output_root" || -L "$output_root" || \
      ! -d "$(dirname "$output_root")" ]]; then
    printf 'finalizer paths are not canonical create-new inputs\n' >&2
    exit 2
fi
for digest in "$run_spec_sha256" "$run_spec_payload_sha256"; do
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
    printf 'union source must be exact clean detached SemTalk with zero branches\n' >&2
    exit 1
fi

orchestrator="$repo_root/scripts/show_base/base_fresh_val_orchestrator.py"
selector="$repo_root/scripts/show_base/select_published_base_winner.py"
if [[ ! -x "$python_bin" || ! -f "$orchestrator" || -L "$orchestrator" || \
      ! -f "$selector" || -L "$selector" ]]; then
    printf 'union implementation is unavailable\n' >&2
    exit 1
fi

partition_paths=()
partition_shas=()
partition_payloads=()
while [[ $# -gt 0 ]]; do
    path=$1
    digest=$2
    payload=$3
    shift 3
    if [[ ! "$path" = /* || ! -f "$path" || -L "$path" || \
          ! "$digest" =~ ^[0-9a-f]{64}$ || \
          ! "$payload" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'unsafe partition receipt triple\n' >&2
        exit 1
    fi
    partition_paths+=("$path")
    partition_shas+=("$digest")
    partition_payloads+=("$payload")
done

mapfile -d '' -t output_root_identity < <(
    "$python_bin" "$orchestrator" create-run-root --path "$output_root" \
        --subdirectory logs
)
if [[ ${#output_root_identity[@]} -ne 3 || \
      "${output_root_identity[0]}" != "$output_root" || \
      ! ${output_root_identity[1]} =~ ^[0-9]+$ || \
      ! ${output_root_identity[2]} =~ ^[0-9]+$ ]]; then
    printf 'finalizer output root could not be created safely\n' >&2
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
    printf 'run-spec finalizer extraction failed\n' >&2
    exit 1
fi
prerequisite=${common[0]}
prerequisite_sha=${common[1]}
prerequisite_bytes=${common[2]}
prerequisite_payload=${common[3]}
continuation=${common[4]}
continuation_sha=${common[5]}
continuation_bytes=${common[6]}
continuation_payload=${common[7]}
real_cache=${common[8]}
real_cache_sha=${common[9]}
real_cache_bytes=${common[10]}
real_cache_payload=${common[11]}
artifact_fields() {
    mapfile -d '' -t artifact_result < <(
        "$python_bin" "$orchestrator" artifact-fields \
            --artifact-path "$1" --payload-receipt
    )
}

union="$output_root/partition-union.json"
union_args=(
    "$python_bin" "$orchestrator" union-partitions
    --partition-count "$partition_count"
)
for ((index = 0; index < partition_count; index++)); do
    union_args+=(
        --partition-receipt "${partition_paths[$index]}"
        --partition-receipt-sha256 "${partition_shas[$index]}"
        --partition-receipt-payload-sha256 "${partition_payloads[$index]}"
    )
done
union_args+=(--output-json "$union")
"${union_args[@]}" >"$output_root/logs/union.log" 2>&1
artifact_fields "$union"
union_sha=${artifact_result[0]}
union_payload=${artifact_result[2]}

evidence="$output_root/candidate-evidence.json"
"$python_bin" "$orchestrator" publish-evidence \
    --partition-union-path "$union" \
    --partition-union-sha256 "$union_sha" \
    --partition-union-payload-sha256 "$union_payload" \
    --prerequisite-path "$prerequisite" \
    --prerequisite-sha256 "$prerequisite_sha" \
    --prerequisite-payload-sha256 "$prerequisite_payload" \
    --continuation-path "$continuation" \
    --continuation-sha256 "$continuation_sha" \
    --continuation-payload-sha256 "$continuation_payload" \
    --real-feature-cache-path "$real_cache" \
    --real-feature-cache-sha256 "$real_cache_sha" \
    --real-feature-cache-payload-sha256 "$real_cache_payload" \
    --output-json "$evidence" >"$output_root/logs/evidence.log" 2>&1
artifact_fields "$evidence"
evidence_sha=${artifact_result[0]}
evidence_bytes=${artifact_result[1]}
evidence_payload=${artifact_result[2]}

winner="$output_root/base-winner-selection.json"
"$python_bin" "$selector" \
    --candidate-evidence-path "$evidence" \
    --candidate-evidence-sha256 "$evidence_sha" \
    --candidate-evidence-bytes "$evidence_bytes" \
    --candidate-evidence-payload-sha256 "$evidence_payload" \
    --prerequisite-selection-path "$prerequisite" \
    --prerequisite-selection-sha256 "$prerequisite_sha" \
    --prerequisite-selection-bytes "$prerequisite_bytes" \
    --prerequisite-selection-payload-sha256 "$prerequisite_payload" \
    --continuation-decision-path "$continuation" \
    --continuation-decision-sha256 "$continuation_sha" \
    --continuation-decision-bytes "$continuation_bytes" \
    --continuation-decision-payload-sha256 "$continuation_payload" \
    --output "$winner" >"$output_root/logs/selection.log" 2>&1

artifact_fields "$winner"
winner_sha=${artifact_result[0]}
winner_payload=${artifact_result[2]}

printf '%s\0%s\0%s\0' "$winner" "$winner_sha" "$winner_payload"
