#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# CPU-only exact union and official 22-way DiffSHEG-FGD selection.  Partition
# receipts are positional by their explicit option names, never discovery.

usage() {
    printf '%s\n' \
        "Usage: $0 --repo-root PATH --python PATH --partition-count 2 --output-root NEW_PATH --source-commit OID --source-tree OID --base-candidate-manifest PATH --expected-base-candidate-manifest-sha256 SHA --base-status-json PATH --expected-base-formal-status-sha256 SHA --base-frozen-inputs-json PATH --expected-base-frozen-inputs-sha256 SHA --val-inputs-json PATH --expected-val-inputs-sha256 SHA --pipeline-json PATH --expected-pipeline-sha256 SHA --partition-0-receipt PATH --expected-partition-0-receipt-sha256 SHA --expected-partition-0-receipt-payload-sha256 SHA --partition-1-receipt PATH --expected-partition-1-receipt-sha256 SHA --expected-partition-1-receipt-payload-sha256 SHA"
}

if ((BASH_VERSINFO[0] < 5)); then
    printf 'bash >= 5 is required\n' >&2
    exit 1
fi

declare -A seen_options=()
repo_root=
python_bin=
partition_count=
output_root=
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
partition_0_receipt=
expected_partition_0_receipt_sha256=
expected_partition_0_receipt_payload_sha256=
partition_1_receipt=
expected_partition_1_receipt_sha256=
expected_partition_1_receipt_payload_sha256=

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
        --partition-count) set_option "$1" partition_count "$2" ;;
        --output-root) set_option "$1" output_root "$2" ;;
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
        --partition-0-receipt)
            set_option "$1" partition_0_receipt "$2" ;;
        --expected-partition-0-receipt-sha256)
            set_option "$1" expected_partition_0_receipt_sha256 "$2" ;;
        --expected-partition-0-receipt-payload-sha256)
            set_option "$1" expected_partition_0_receipt_payload_sha256 "$2" ;;
        --partition-1-receipt)
            set_option "$1" partition_1_receipt "$2" ;;
        --expected-partition-1-receipt-sha256)
            set_option "$1" expected_partition_1_receipt_sha256 "$2" ;;
        --expected-partition-1-receipt-payload-sha256)
            set_option "$1" expected_partition_1_receipt_payload_sha256 "$2" ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift 2
done

for required in \
    repo_root python_bin partition_count output_root source_commit source_tree \
    base_candidate_manifest expected_base_candidate_manifest_sha256 \
    base_status_json expected_base_formal_status_sha256 \
    base_frozen_inputs_json expected_base_frozen_inputs_sha256 \
    val_inputs_json expected_val_inputs_sha256 pipeline_json \
    expected_pipeline_sha256 partition_0_receipt \
    expected_partition_0_receipt_sha256 \
    expected_partition_0_receipt_payload_sha256 partition_1_receipt \
    expected_partition_1_receipt_sha256 \
    expected_partition_1_receipt_payload_sha256; do
    if [[ -z ${!required} ]]; then
        printf 'missing required option value: %s\n' "$required" >&2
        exit 2
    fi
done

# Fail before consuming a new-only destination.
if [[ $partition_count != 2 ]]; then
    printf 'formal union requires exactly two partitions\n' >&2
    exit 2
fi
for digest in \
    "$expected_base_candidate_manifest_sha256" \
    "$expected_base_formal_status_sha256" \
    "$expected_base_frozen_inputs_sha256" "$expected_val_inputs_sha256" \
    "$expected_pipeline_sha256" "$expected_partition_0_receipt_sha256" \
    "$expected_partition_0_receipt_payload_sha256" \
    "$expected_partition_1_receipt_sha256" \
    "$expected_partition_1_receipt_payload_sha256"; do
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
    "$repo_root" "$python_bin" "$output_root" "$base_candidate_manifest" \
    "$base_status_json" "$base_frozen_inputs_json" "$val_inputs_json" \
    "$pipeline_json" "$partition_0_receipt" "$partition_1_receipt"; do
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
output_parent=$(realpath -e -- "$(dirname -- "$output_root")")
if [[ $output_root != "$output_parent/$(basename -- "$output_root")" || \
      -e $output_root || -L $output_root ]]; then
    printf 'final output root must be one new canonical path\n' >&2
    exit 1
fi
case "$output_root/" in
    "$repo_root/"*)
        printf 'formal output must remain outside the source checkout\n' >&2
        exit 1
        ;;
esac

finalizer_name=finalize_base_diffsheg_val_partitions.sh
if [[ -L ${BASH_SOURCE[0]} ]]; then
    printf 'formal finalizer must not be a symlink\n' >&2
    exit 1
fi
finalizer_path=$(realpath -e -- "${BASH_SOURCE[0]}")
launcher_dir=$repo_root/scripts/show_base
if [[ $finalizer_path != "$launcher_dir/$finalizer_name" ]]; then
    printf 'finalizer is not the tracked entrypoint in the pinned source\n' >&2
    exit 1
fi
partition_contract=$launcher_dir/base_diffsheg_val_partition_contract.py
selector=$launcher_dir/select_base_official_adapt_long.py
python_runtime_contract=$launcher_dir/formal_python_runtime_contract.sh
for required_path in \
    "$partition_contract" "$selector" "$python_runtime_contract"; do
    if [[ ! -f $required_path || -L $required_path ]]; then
        printf 'required tracked source is unavailable: %s\n' "$required_path" >&2
        exit 1
    fi
done
if [[ $(git -C "$repo_root" remote get-url origin) != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" rev-parse HEAD) != "$source_commit" || \
      $(git -C "$repo_root" rev-parse 'HEAD^{tree}') != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf 'union source must be the exact clean detached zero-branch checkout\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/finalize_base_diffsheg_val_partitions.sh \
    scripts/show_base/formal_python_runtime_contract.sh \
    scripts/show_base/base_diffsheg_val_partition_contract.py \
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
    --path "$output_root" --subdirectory logs >/dev/null

artifact_fields() {
    local path=$1
    mapfile -d '' -t artifact_result < <(
        "$python_bin" "$partition_contract" artifact-fields \
            --artifact-path "$path" --payload-receipt
    )
    if [[ ${#artifact_result[@]} -ne 3 || \
          ! ${artifact_result[0]} =~ ^[0-9a-f]{64}$ || \
          ! ${artifact_result[2]} =~ ^[0-9a-f]{64}$ ]]; then
        printf 'cannot pin output artifact: %s\n' "$path" >&2
        exit 1
    fi
}

union=$output_root/partition-union.json
"$python_bin" "$partition_contract" union-partitions \
    --partition-count "$partition_count" \
    --source-commit "$source_commit" --source-tree "$source_tree" \
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
    --partition-receipt "$partition_0_receipt" \
    --expected-partition-receipt-sha256 \
        "$expected_partition_0_receipt_sha256" \
    --expected-partition-receipt-payload-sha256 \
        "$expected_partition_0_receipt_payload_sha256" \
    --partition-receipt "$partition_1_receipt" \
    --expected-partition-receipt-sha256 \
        "$expected_partition_1_receipt_sha256" \
    --expected-partition-receipt-payload-sha256 \
        "$expected_partition_1_receipt_payload_sha256" \
    --output-json "$union" >"$output_root/logs/union.log" 2>&1
artifact_fields "$union"
union_sha=${artifact_result[0]}
union_payload=${artifact_result[2]}

mapfile -d '' -t measurement_fields < <(
    "$python_bin" "$partition_contract" extract-union \
        --union-json "$union" --expected-union-sha256 "$union_sha" \
        --expected-union-payload-sha256 "$union_payload"
)
if [[ ${#measurement_fields[@]} -ne 44 ]]; then
    printf 'union did not expose exactly 22 measurement path/SHA pairs\n' >&2
    exit 1
fi

selection=$output_root/base-winner-selection.json
selector_command=(
    "$python_bin" "$selector"
    --base-candidate-manifest "$base_candidate_manifest"
    --expected-base-candidate-manifest-sha256
        "$expected_base_candidate_manifest_sha256"
    --base-status-json "$base_status_json"
    --expected-base-formal-status-sha256
        "$expected_base_formal_status_sha256"
    --base-frozen-inputs-json "$base_frozen_inputs_json"
    --expected-base-frozen-inputs-sha256
        "$expected_base_frozen_inputs_sha256"
)
for ((index = 0; index < 44; index += 2)); do
    selector_command+=(
        --measurement-json "${measurement_fields[$index]}"
        --expected-measurement-sha256 "${measurement_fields[$((index + 1))]}"
    )
done
selector_command+=(--output-json "$selection")
"${selector_command[@]}" >"$output_root/logs/selection.log" 2>&1

artifact_fields "$selection"
printf '%s\0%s\0%s\0' \
    "$selection" "${artifact_result[0]}" "${artifact_result[2]}"
