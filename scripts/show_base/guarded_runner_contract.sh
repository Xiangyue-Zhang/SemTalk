#!/usr/bin/env bash
# Shared fail-closed parent contract for every formal SHOW GPU launcher.
#
# This file is sourced by launchers; it never starts work on its own.  The
# public function accepts no caller-supplied identity.  It derives the direct
# parent from Bash's read-only PPID, snapshots that Linux /proc identity, and
# accepts exactly one guarded-runner script argv element plus exactly one
# all-GPU reservation.

_semtalk_guarded_runner_argv_is_exact() {
    local python_name
    local runner_path_count=0
    local delimiter_index=-1
    local gpu_option_count=0
    local exact_gpu_count=0
    local index
    local -a argv=("$@")

    # The runner is a Python script.  Whether invoked through its shebang or
    # an explicit interpreter, Linux presents the interpreter as argv[0] and
    # the actual script as argv[1].  Tokens in workload argv after the first
    # runner `--` separator are deliberately not runner authority.
    (( ${#argv[@]} >= 5 )) || return 1
    python_name=${argv[0]##*/}
    [[ "$python_name" =~ ^python([0-9]+([.][0-9]+)*)?$ && \
       ${argv[1]} == /tmp/globaldiff_guarded_runner.py ]] || return 1
    # Only the first separator belongs to the guarded runner.  The workload
    # argv after it is inert authority and may contain any number of its own
    # ``--`` separators.
    for ((index = 2; index < ${#argv[@]}; index++)); do
        if [[ ${argv[$index]} == -- ]]; then
            delimiter_index=$index
            break
        fi
    done
    [[ "$delimiter_index" -gt 2 && \
       "$delimiter_index" -lt $((${#argv[@]} - 1)) ]] || return 1

    for ((index = 2; index < delimiter_index; index++)); do
        if [[ ${argv[$index]} == /tmp/globaldiff_guarded_runner.py ]]; then
            ((runner_path_count += 1))
        fi
        if [[ ${argv[$index]} == --gpus ]]; then
            ((gpu_option_count += 1))
            if [[ $((index + 1)) -lt "$delimiter_index" && \
                  ${argv[$((index + 1))]} == 0,1,2,3,4,5,6,7 ]]; then
                ((exact_gpu_count += 1))
            fi
        elif [[ ${argv[$index]} == --gpus=* ]]; then
            ((gpu_option_count += 1))
            if [[ ${argv[$index]} == --gpus=0,1,2,3,4,5,6,7 ]]; then
                ((exact_gpu_count += 1))
            fi
        fi
    done

    # argv[1] is the one authoritative runner path; it must not be repeated
    # among runner options.  Workload tokens after `--` are outside this count.
    ((runner_path_count += 1))
    [[ "$runner_path_count" -eq 1 && \
       "$gpu_option_count" -eq 1 && \
       "$exact_gpu_count" -eq 1 ]]
}

_semtalk_guarded_runner_proc_identity() {
    local pid=$1
    local stat_line stat_tail
    local -a stat_fields

    [[ "$pid" =~ ^[1-9][0-9]*$ && \
       -r "/proc/$pid/stat" && \
       -r "/proc/$pid/cmdline" ]] || return 1
    stat_line=$(<"/proc/$pid/stat") || return 1
    [[ "$stat_line" == *") "* ]] || return 1
    stat_tail=${stat_line##*) }
    read -r -a stat_fields <<<"$stat_tail"
    (( ${#stat_fields[@]} >= 20 )) || return 1
    [[ ${stat_fields[0]} != Z && ${stat_fields[19]} =~ ^[0-9]+$ ]] || return 1
    # Process state may legitimately move between sleeping/runnable while the
    # snapshot is taken.  PID plus immutable Linux starttime is the identity;
    # state is checked only to reject a zombie.
    printf '%s\n' "${stat_fields[19]}"
}

semtalk_require_exact_guarded_runner_all_gpus() {
    local runner_pid=$PPID
    local identity_before identity_after
    local -a runner_argv

    identity_before=$(_semtalk_guarded_runner_proc_identity "$runner_pid") || {
        printf 'guarded runner process is unavailable: %s\n' \
            "$runner_pid" >&2
        return 1
    }
    mapfile -d '' -t runner_argv <"/proc/$runner_pid/cmdline" || {
        printf 'cannot snapshot guarded runner argv: %s\n' \
            "$runner_pid" >&2
        return 1
    }
    identity_after=$(_semtalk_guarded_runner_proc_identity "$runner_pid") || {
        printf 'guarded runner exited during argv snapshot: %s\n' \
            "$runner_pid" >&2
        return 1
    }
    if [[ "$identity_before" != "$identity_after" ]]; then
        printf 'guarded runner identity changed during argv snapshot: %s\n' \
            "$runner_pid" >&2
        return 1
    fi
    if ! _semtalk_guarded_runner_argv_is_exact "${runner_argv[@]}"; then
        printf '%s\n' \
            'requires direct parent /tmp/globaldiff_guarded_runner.py with' \
            'one exact --gpus 0,1,2,3,4,5,6,7 reservation' >&2
        return 1
    fi
}
