#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf '%s\n' \
        "Usage: $0 PYTHON [train_base_official_adapt_long.py arguments...]" >&2
    exit 2
fi

python_bin=$1
shift
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

if [[ ! -x "$python_bin" ]]; then
    printf 'Python executable is missing or not executable: %s\n' \
        "$python_bin" >&2
    exit 2
fi

# PYTHONHASHSEED is read only at interpreter startup.  Bind it here, before
# torchrun creates the eight formal rank interpreters; the trainer independently
# verifies the value and fails closed if it was bypassed.
export PYTHONHASHSEED=43
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec "$python_bin" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    "$script_dir/train_base_official_adapt_long.py" \
    "$@"
