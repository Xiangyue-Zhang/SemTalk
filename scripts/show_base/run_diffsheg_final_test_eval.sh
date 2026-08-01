#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# This compatibility wrapper is preflight-only.  The non-retryable formal
# event has exactly one entrypoint: run_base_final_test.sh, which owns the
# guarded child, pin FIFO, and mandatory terminal validator receipt.

usage() {
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON SOURCE_COMMIT SOURCE_TREE -- [evaluate_diffsheg_final_test.py arguments...]"
}

if [[ $# -lt 6 || ${5-} != -- ]]; then
    usage >&2
    exit 2
fi

repo_root=$1
python_bin=$2
source_commit=$3
source_tree=$4
shift 5

if [[ $repo_root != /* || $python_bin != /* || \
      ! $source_commit =~ ^[0-9a-f]{40}$ || \
      ! $source_tree =~ ^[0-9a-f]{40}$ ]]; then
    printf '%s\n' 'formal source paths/OIDs are invalid' >&2
    exit 2
fi

raw_repo_root=$repo_root
repo_root=$(realpath -- "$repo_root")
if [[ $repo_root != "$raw_repo_root" || ! -d $repo_root || \
      -L $raw_repo_root ]]; then
    printf '%s\n' 'repository root must be canonical and non-symlinked' >&2
    exit 1
fi
if [[ ! -f $python_bin || ! -x $python_bin ]]; then
    printf '%s\n' 'formal Python is unavailable' >&2
    exit 1
fi

launcher_name=run_diffsheg_final_test_eval.sh
if [[ -L ${BASH_SOURCE[0]} ]]; then
    printf '%s\n' 'formal launcher must not be a symlink' >&2
    exit 1
fi
launcher_path=$(realpath -- "${BASH_SOURCE[0]}")
launcher_dir=$repo_root/scripts/show_base
if [[ $launcher_path != "$launcher_dir/$launcher_name" ]]; then
    printf '%s\n' 'launcher is not the tracked pinned-source entrypoint' >&2
    exit 1
fi

entrypoint=$launcher_dir/evaluate_diffsheg_final_test.py
python_runtime_contract=$launcher_dir/formal_python_runtime_contract.sh
for required in "$entrypoint" "$python_runtime_contract"; do
    if [[ ! -f $required || -L $required ]]; then
        printf 'required formal source is missing or unsafe: %s\n' \
            "$required" >&2
        exit 1
    fi
done

if [[ $(git -C "$repo_root" remote get-url origin) != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" remote get-url --push origin) != \
      git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" rev-parse HEAD) != "$source_commit" || \
      $(git -C "$repo_root" rev-parse 'HEAD^{tree}') != "$source_tree" || \
      -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" || \
      -n "$(git -C "$repo_root" symbolic-ref -q --short HEAD || true)" || \
      -n "$(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads)" ]]; then
    printf '%s\n' \
        'formal source must be exact clean detached SemTalk with zero branches' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/run_diffsheg_final_test_eval.sh \
    scripts/show_base/evaluate_diffsheg_final_test.py \
    scripts/show_base/formal_python_runtime_contract.sh; do
    if [[ $(git -C "$repo_root" ls-files --error-unmatch "$tracked") != \
          "$tracked" ]]; then
        printf 'formal source is not tracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

preflight_count=0
for argument in "$@"; do
    case $argument in
        --preflight-only)
            ((preflight_count += 1))
            ;;
        --preflight-only=*)
            printf '%s\n' '--preflight-only is a valueless exact flag' >&2
            exit 2
            ;;
        --skip-ba|--skip-ba=*|\
        --validate-input-only|--validate-input-only=*|\
        --window-stride|--window-stride=*|\
        --audio-dir|--audio-dir=*|\
        *speaker2*|*Speaker2*)
            printf 'forbidden formal DiffSHEG argument/label: %s\n' \
                "$argument" >&2
            exit 2
            ;;
    esac
done
if ((preflight_count > 1)); then
    printf '%s\n' 'duplicate --preflight-only flag' >&2
    exit 2
fi

if ((preflight_count != 1)); then
    printf '%s\n' \
        'formal final metrics must run only through run_base_final_test.sh' >&2
    exit 2
fi

. "$python_runtime_contract"
semtalk_require_formal_venv_python "$python_bin" semtalk

exec "$python_bin" "$entrypoint" "$@"
