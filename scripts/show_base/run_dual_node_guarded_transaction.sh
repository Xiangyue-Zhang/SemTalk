#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Generic two-node transaction launcher.  This file must itself be the direct
# workload of /tmp/globaldiff_guarded_runner.py with the exact all-GPU
# reservation.  It validates that parent and then execs the Python coordinator
# so the coordinator's PPID remains the guarded runner for immutable evidence.

usage() {
    printf '%s\n' \
        "Usage: $0 PYTHON [dual_node_guarded_transaction.py options] -- /absolute/workload [args ...]"
}

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi

python_bin=$1
shift
launcher_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
helper="$launcher_dir/dual_node_guarded_transaction.py"

# shellcheck source=guarded_runner_contract.sh
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

if [[ ! "$python_bin" = /* || ! -x "$python_bin" || -L "$python_bin" ]]; then
    printf '%s\n' 'Python must be an absolute executable non-symlink file' >&2
    exit 2
fi
if [[ ! -f "$helper" || -L "$helper" ]]; then
    printf '%s\n' 'transaction helper is missing or unsafe' >&2
    exit 1
fi

exec "$python_bin" "$helper" "$@"
