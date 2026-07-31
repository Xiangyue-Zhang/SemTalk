#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Formal mode is a GPU task and this launcher must itself be invoked beneath:
#   /tmp/globaldiff_guarded_runner.py -- ...
# The CPU-only --preflight-only mode may be run directly.

if [[ $# -lt 3 ]]; then
    printf '%s\n' \
        "Usage: $0 REPO_ROOT PYTHON [evaluate_diffsheg_final_test.py arguments...]" >&2
    exit 2
fi

repo_root=$1
python_bin=$2
shift 2
entrypoint="$repo_root/scripts/show_base/evaluate_diffsheg_final_test.py"

for required in "$repo_root" "$python_bin" "$entrypoint"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 1
    fi
done

for argument in "$@"; do
    case "$argument" in
        --skip-ba|--skip-ba=*|\
        --validate-input-only|--validate-input-only=*|\
        --window-stride|--window-stride=*|\
        --audio-dir|--audio-dir=*|\
        *released2*|*paper16*|*speaker2*|*Speaker2*)
            printf 'forbidden formal DiffSHEG argument/label: %s\n' \
                "$argument" >&2
            exit 2
            ;;
    esac
done

exec "$python_bin" "$entrypoint" "$@"
