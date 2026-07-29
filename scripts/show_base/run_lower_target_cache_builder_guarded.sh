#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# This bootstrap must be the unique direct workload of
# /tmp/globaldiff_guarded_runner.py --gpus 0,1,2,3,4,5,6,7.  It resolves the
# otherwise self-referential runner PID/starttime/argv hash from $PPID, then
# execs the standard-library process wrapper in place (same PID).

if (( $# < 2 )); then
    printf '%s\n' \
        "Usage: $0 PYTHON REPO_ROOT [run_lower_target_cache_builder.py arguments]" \
        >&2
    exit 2
fi

python_bin=$1
repo_root=$2
shift 2
wrapper=$repo_root/scripts/show_base/run_lower_target_cache_builder.py

if [[ ! -x "$python_bin" || ! -f "$wrapper" || -L "$wrapper" ]]; then
    printf 'invalid builder bootstrap input\n' >&2
    exit 1
fi
for argument in "$@"; do
    case "$argument" in
        --expected-runner-pid|--expected-runner-starttime|\
        --expected-runner-argv-sha256)
            printf 'runner identity arguments are bootstrap-owned\n' >&2
            exit 2
            ;;
    esac
done

runner_pid=$PPID
if [[ ! -r "/proc/$runner_pid/stat" || \
      ! -r "/proc/$runner_pid/cmdline" ]]; then
    printf 'guarded runner process is unavailable: %s\n' "$runner_pid" >&2
    exit 1
fi
stat_line=$(<"/proc/$runner_pid/stat")
[[ "$stat_line" == *") "* ]] || {
    printf 'invalid guarded runner stat record\n' >&2
    exit 1
}
stat_tail=${stat_line##*) }
read -r -a stat_fields <<<"$stat_tail"
(( ${#stat_fields[@]} > 19 )) || {
    printf 'incomplete guarded runner stat record\n' >&2
    exit 1
}
runner_starttime=${stat_fields[19]}
runner_argv_sha=$(sha256sum "/proc/$runner_pid/cmdline")
runner_argv_sha=${runner_argv_sha%% *}

mapfile -d '' -t runner_argv <"/proc/$runner_pid/cmdline"
runner_path_ok=false
runner_gpus_ok=false
for ((index = 0; index < ${#runner_argv[@]}; index++)); do
    if [[ ${runner_argv[$index]} == /tmp/globaldiff_guarded_runner.py ]]; then
        runner_path_ok=true
    fi
    if [[ ${runner_argv[$index]} == --gpus && \
          $((index + 1)) -lt ${#runner_argv[@]} && \
          ${runner_argv[$((index + 1))]} == 0,1,2,3,4,5,6,7 ]]; then
        runner_gpus_ok=true
    fi
    if [[ ${runner_argv[$index]} == \
          --gpus=0,1,2,3,4,5,6,7 ]]; then
        runner_gpus_ok=true
    fi
done
if [[ "$runner_path_ok" != true || "$runner_gpus_ok" != true ]]; then
    printf 'builder requires exact guarded-runner all-GPU reservation\n' >&2
    exit 1
fi

exec "$python_bin" "$wrapper" "$@" \
    --expected-runner-pid "$runner_pid" \
    --expected-runner-starttime "$runner_starttime" \
    --expected-runner-argv-sha256 "$runner_argv_sha"
