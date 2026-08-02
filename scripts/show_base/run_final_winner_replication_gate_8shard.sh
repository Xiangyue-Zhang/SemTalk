#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Formal final-winner deterministic replication transaction.  This launcher
# must be the direct workload child of /tmp/globaldiff_guarded_runner.py.  It
# runs seeds 0 and 15 as two independent eight-shard waves, finalizes each
# seed in a separate process, invokes the sole canonical gate finalizer, and
# emits one completion receipt binding the gate to the validation winner.

usage() {
    printf '%s\n' \
        "Usage: $0 --repo-root PATH --python PATH --preflight PATH --expected-preflight-sha256 SHA --winner-selection PATH --expected-winner-selection-sha256 SHA --run-root NEW_PATH --source-commit OID --source-tree OID --expected-guarded-runner-sha256 SHA"
}

if (($# == 1)) && [[ $1 == --help ]]; then
    usage
    exit 0
fi

if ((BASH_VERSINFO[0] < 5)); then
    printf 'bash >= 5 is required\n' >&2
    exit 1
fi

declare -A seen_options=()
repo_root=
python_bin=
preflight=
expected_preflight_sha256=
winner_selection=
expected_winner_selection_sha256=
run_root=
source_commit=
source_tree=
expected_guarded_runner_sha256=

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
        --preflight) set_option "$1" preflight "$2" ;;
        --expected-preflight-sha256)
            set_option "$1" expected_preflight_sha256 "$2" ;;
        --winner-selection) set_option "$1" winner_selection "$2" ;;
        --expected-winner-selection-sha256)
            set_option "$1" expected_winner_selection_sha256 "$2" ;;
        --run-root) set_option "$1" run_root "$2" ;;
        --source-commit) set_option "$1" source_commit "$2" ;;
        --source-tree) set_option "$1" source_tree "$2" ;;
        --expected-guarded-runner-sha256)
            set_option "$1" expected_guarded_runner_sha256 "$2" ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift 2
done

for required in \
    repo_root python_bin preflight expected_preflight_sha256 \
    winner_selection expected_winner_selection_sha256 run_root \
    source_commit source_tree expected_guarded_runner_sha256; do
    if [[ -z ${!required} ]]; then
        printf 'missing required option: %s\n' "$required" >&2
        exit 2
    fi
done
for digest in \
    "$expected_preflight_sha256" \
    "$expected_winner_selection_sha256" \
    "$expected_guarded_runner_sha256"; do
    [[ $digest =~ ^[0-9a-f]{64}$ ]] || {
        printf 'invalid SHA-256 argument\n' >&2
        exit 2
    }
done
for oid in "$source_commit" "$source_tree"; do
    [[ $oid =~ ^[0-9a-f]{40}$ ]] || {
        printf 'invalid Git object ID argument\n' >&2
        exit 2
    }
done
for path_value in \
    "$repo_root" "$python_bin" "$preflight" "$winner_selection" \
    "$run_root"; do
    [[ $path_value == /* ]] || {
        printf 'all replication paths must be absolute\n' >&2
        exit 2
    }
done

raw_repo_root=$repo_root
repo_root=$(realpath -e -- "$repo_root")
if [[ $repo_root != "$raw_repo_root" || ! -d $repo_root || -L $raw_repo_root ]]; then
    printf 'repository root must be canonical and non-symlinked\n' >&2
    exit 1
fi
for input in "$preflight" "$winner_selection"; do
    if [[ ! -f $input || -L $input || $(realpath -e -- "$input") != "$input" ]]; then
        printf 'replication input must be a canonical regular file: %s\n' \
            "$input" >&2
        exit 1
    fi
done
run_parent=$(realpath -e -- "$(dirname -- "$run_root")")
if [[ $run_root != "$run_parent/$(basename -- "$run_root")" || \
      -e $run_root || -L $run_root ]]; then
    printf 'run root must be one absent canonical path\n' >&2
    exit 1
fi
case "$run_root/" in
    "$repo_root/"*)
        printf 'run root must remain outside the source checkout\n' >&2
        exit 1
        ;;
esac

launcher_dir=$repo_root/scripts/show_base
producer=$launcher_dir/produce_final_winner_replication_seed.py
gate_finalizer=$launcher_dir/deterministic_replication_gate.py
guard_contract=$launcher_dir/guarded_runner_contract.sh
python_contract=$launcher_dir/formal_python_runtime_contract.sh
launcher=$launcher_dir/run_final_winner_replication_gate_8shard.sh
for required_path in \
    "$producer" "$gate_finalizer" "$guard_contract" "$python_contract" \
    "$launcher"; do
    if [[ ! -f $required_path || -L $required_path ]]; then
        printf 'required replication source is unavailable: %s\n' \
            "$required_path" >&2
        exit 1
    fi
done
if [[ $(realpath -e -- "${BASH_SOURCE[0]}") != "$launcher" ]]; then
    printf 'launcher is not the tracked source entrypoint\n' >&2
    exit 1
fi

# shellcheck source=guarded_runner_contract.sh
. "$guard_contract"
semtalk_require_exact_guarded_runner_all_gpus

guarded_runner=/tmp/globaldiff_guarded_runner.py
if [[ ! -f $guarded_runner || -L $guarded_runner || \
      $(realpath -e -- "$guarded_runner") != "$guarded_runner" ]]; then
    printf 'guarded runner is unavailable or unsafe\n' >&2
    exit 1
fi
observed_guarded_runner_sha256=$(sha256sum -- "$guarded_runner")
observed_guarded_runner_sha256=${observed_guarded_runner_sha256%% *}
if [[ $observed_guarded_runner_sha256 != \
      "$expected_guarded_runner_sha256" ]]; then
    printf 'guarded runner SHA-256 mismatch\n' >&2
    exit 1
fi

if [[ $(git -C "$repo_root" remote) != origin || \
      $(git -C "$repo_root" remote get-url origin) != \
          git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" remote get-url --push origin) != \
          git@github.com:Xiangyue-Zhang/SemTalk.git || \
      $(git -C "$repo_root" rev-parse HEAD) != "$source_commit" || \
      $(git -C "$repo_root" rev-parse 'HEAD^{tree}') != "$source_tree" || \
      -n $(git -C "$repo_root" status --porcelain=v1 --untracked-files=all) || \
      -n $(git -C "$repo_root" symbolic-ref -q --short HEAD || true) || \
      -n $(git -C "$repo_root" for-each-ref --format='%(refname)' refs/heads) ]]; then
    printf 'replication source must be exact clean detached zero-branch SemTalk\n' >&2
    exit 1
fi
for tracked in \
    scripts/show_base/produce_final_winner_replication_seed.py \
    scripts/show_base/run_final_winner_replication_gate_8shard.sh \
    scripts/show_base/deterministic_replication_gate.py \
    scripts/show_base/run_base_val_inference.py \
    scripts/show_base/semtalk_base_inference_core.py \
    scripts/show_base/guarded_runner_contract.sh \
    scripts/show_base/formal_python_runtime_contract.sh \
    scripts/show_base/validate_base_long_test_winner.py \
    scripts/show_base/base_long_val_contract.py; do
    if [[ $(git -C "$repo_root" ls-files --error-unmatch "$tracked") != \
          "$tracked" ]]; then
        printf 'replication source is not tracked: %s\n' "$tracked" >&2
        exit 1
    fi
done

# shellcheck source=formal_python_runtime_contract.sh
. "$python_contract"
semtalk_require_formal_venv_python "$python_bin" semtalk

mkdir -- "$run_root"
mkdir -- "$run_root/logs"
cd "$repo_root"

authority_root=$run_root/control
"$python_bin" "$producer" prepare \
    --repo-root "$repo_root" \
    --preflight "$preflight" \
    --expected-preflight-sha256 "$expected_preflight_sha256" \
    --winner-selection "$winner_selection" \
    --expected-winner-selection-sha256 \
        "$expected_winner_selection_sha256" \
    --source-commit "$source_commit" --source-tree "$source_tree" \
    --output-root "$authority_root" \
    >"$run_root/logs/prepare.log" 2>&1
authority=$authority_root/replication-authority.json
authority_sha256=$(sha256sum -- "$authority")
authority_sha256=${authority_sha256%% *}

declare -a active_pids=()
declare -a active_starts=()
declare -a active_argv_sha256=()

proc_start() {
    local line rest
    local -a fields
    line=$(<"/proc/$1/stat") || return 1
    [[ $line == *") "* ]] || return 1
    rest=${line##*) }
    read -r -a fields <<<"$rest"
    [[ ${fields[0]} != Z && ${fields[19]} =~ ^[0-9]+$ ]] || return 1
    printf '%s' "${fields[19]}"
}

capture_child() {
    local pid=$1 expected_argv_sha=$2 observed_parent observed_start
    for _attempt in {1..400}; do
        if [[ -r /proc/$pid/status && -r /proc/$pid/stat && \
              -r /proc/$pid/cmdline ]]; then
            observed_parent=$(awk '/^PPid:/{print $2}' "/proc/$pid/status")
            observed_start=$(proc_start "$pid" || true)
            if [[ $observed_parent == "$BASHPID" && \
                  $observed_start =~ ^[0-9]+$ ]]; then
                observed_argv_sha=$(sha256sum -- "/proc/$pid/cmdline")
                observed_argv_sha=${observed_argv_sha%% *}
                [[ $observed_argv_sha == "$expected_argv_sha" ]] || return 1
                active_pids+=("$pid")
                active_starts+=("$observed_start")
                active_argv_sha256+=("$observed_argv_sha")
                return 0
            fi
        else
            return 1
        fi
        sleep 0.01
    done
    return 1
}

cleanup_exact_children() {
    local index pid expected_start observed_parent observed_start
    for index in "${!active_pids[@]}"; do
        pid=${active_pids[$index]}
        expected_start=${active_starts[$index]}
        if [[ -r /proc/$pid/status && -r /proc/$pid/stat ]]; then
            observed_parent=$(awk '/^PPid:/{print $2}' "/proc/$pid/status")
            observed_start=$(proc_start "$pid" || true)
            if [[ $observed_parent == "$BASHPID" && \
                  $observed_start == "$expected_start" ]]; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        fi
    done
}

cleanup_on_signal() {
    local signal_number=$1
    cleanup_exact_children
    trap - EXIT INT TERM
    exit "$signal_number"
}

trap cleanup_exact_children EXIT
trap 'cleanup_on_signal 130' INT
trap 'cleanup_on_signal 143' TERM

run_seed() {
    local seed=$1 shard_id pid status=0 argv_sha
    local seed_root=$run_root/seed-$seed
    mkdir -- "$seed_root"
    active_pids=()
    active_starts=()
    active_argv_sha256=()
    for shard_id in {0..7}; do
        local -a command=(
            "$python_bin" "$producer" shard
            --authority "$authority"
            --expected-authority-sha256 "$authority_sha256"
            --seed-root "$seed_root"
            --seed "$seed"
            --num-shards 8
            --shard-id "$shard_id"
            --device "cuda:$shard_id"
            --progress-every 20
        )
        argv_sha=$(printf '%s\0' "${command[@]}" | sha256sum)
        argv_sha=${argv_sha%% *}
        "${command[@]}" \
            >"$run_root/logs/seed-$seed-shard-$shard_id.log" 2>&1 &
        pid=$!
        if ! capture_child "$pid" "$argv_sha"; then
            printf 'cannot capture exact seed %s shard %s process\n' \
                "$seed" "$shard_id" >&2
            wait "$pid" 2>/dev/null || true
            return 1
        fi
    done
    for pid in "${active_pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    active_pids=()
    active_starts=()
    active_argv_sha256=()
    if ((status != 0)); then
        printf 'seed %s shard wave failed\n' "$seed" >&2
        return 1
    fi
    "$python_bin" "$producer" finalize-seed \
        --authority "$authority" \
        --expected-authority-sha256 "$authority_sha256" \
        --seed-root "$seed_root" --seed "$seed" \
        --output-json "$run_root/seed-$seed-receipt.json" \
        >"$run_root/logs/seed-$seed-finalize.log" 2>&1
}

run_seed 0
run_seed 15

seed0=$run_root/seed-0-receipt.json
seed15=$run_root/seed-15-receipt.json
seed0_sha=$(sha256sum -- "$seed0")
seed0_sha=${seed0_sha%% *}
seed15_sha=$(sha256sum -- "$seed15")
seed15_sha=${seed15_sha%% *}
gate_json=$run_root/final-winner-replication-gate.json
"$python_bin" "$gate_finalizer" finalize \
    --seed-run-json "$seed0" \
    --expected-seed-run-sha256 "$seed0_sha" \
    --seed-run-json "$seed15" \
    --expected-seed-run-sha256 "$seed15_sha" \
    --scope final_winner --output-json "$gate_json" \
    >"$run_root/logs/gate-finalize.log" 2>&1
gate_sha=$(sha256sum -- "$gate_json")
gate_sha=${gate_sha%% *}

completion=$run_root/completion.json
"$python_bin" "$producer" complete \
    --authority "$authority" \
    --expected-authority-sha256 "$authority_sha256" \
    --gate-json "$gate_json" --expected-gate-sha256 "$gate_sha" \
    --guarded-runner-sha256 "$expected_guarded_runner_sha256" \
    --output-json "$completion" \
    >"$run_root/logs/complete.log" 2>&1

trap - EXIT INT TERM
completion_sha=$(sha256sum -- "$completion")
completion_sha=${completion_sha%% *}
printf '{"completion":"%s","sha256":"%s","status":"complete"}\n' \
    "$completion" "$completion_sha"
