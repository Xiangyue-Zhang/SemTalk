#!/usr/bin/env bash

# Pure launcher-policy helpers.  Keep these free of filesystem and process
# side effects so the exact partial/full scheduling contract can be executed
# in CPU-only tests.

semtalk_prereq_candidate_index_authority() {
    if [[ $# -ne 3 ]]; then
        printf 'candidate-index authority requires actual, legacy-partial, and segmented-partial formats\n' >&2
        return 2
    fi

    local actual_format=$1
    local legacy_partial_format=$2
    local segmented_partial_format=$3
    if [[ "$actual_format" == "$legacy_partial_format" || \
          "$actual_format" == "$segmented_partial_format" ]]; then
        printf 'partial\t40\t4\tface\n'
    else
        printf 'complete\t50\t5\tglobal\n'
    fi
}
