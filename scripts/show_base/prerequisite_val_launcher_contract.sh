#!/usr/bin/env bash

# Pure launcher-policy helpers.  Keep these free of filesystem and process
# side effects so the exact partial/full scheduling contract can be executed
# in CPU-only tests.

semtalk_prereq_candidate_index_authority() {
    if [[ $# -ne 5 ]]; then
        printf 'candidate-index authority requires actual plus all four legacy/segmented partial/complete formats\n' >&2
        return 2
    fi

    local actual_format=$1
    local legacy_partial_format=$2
    local segmented_partial_format=$3
    local legacy_complete_format=$4
    local segmented_complete_format=$5
    if [[ "$actual_format" == "$legacy_partial_format" ]]; then
        printf 'partial\t40\t4\tface\n'
    elif [[ "$actual_format" == "$segmented_partial_format" ]]; then
        printf 'partial\t40\t0\tface\n'
    elif [[ "$actual_format" == "$legacy_complete_format" ]]; then
        printf 'complete\t50\t5\tglobal\n'
    elif [[ "$actual_format" == "$segmented_complete_format" ]]; then
        printf 'complete\t50\t0\tglobal\n'
    else
        printf 'candidate-index authority format is unknown\n' >&2
        return 1
    fi
}

semtalk_prereq_plan_job_count_valid() {
    if [[ $# -ne 3 ]]; then
        printf 'plan-job validation requires actual jobs, minimum jobs, and legacy modulus\n' >&2
        return 2
    fi

    local actual_jobs=$1
    local minimum_jobs=$2
    local legacy_modulus=$3
    if [[ ! "$actual_jobs" =~ ^(0|[1-9][0-9]*)$ || \
          ! "$minimum_jobs" =~ ^[1-9][0-9]*$ || \
          ! "$legacy_modulus" =~ ^(0|[1-9][0-9]*)$ ]]; then
        printf 'plan-job validation arguments are not canonical integers\n' >&2
        return 2
    fi
    if ((actual_jobs < minimum_jobs)); then
        return 1
    fi
    if ((legacy_modulus > 0 && actual_jobs % legacy_modulus != 0)); then
        return 1
    fi
    return 0
}
