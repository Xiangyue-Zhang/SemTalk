#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    printf '%s\n' \
        "Usage: $0 PYTHON TOPOLOGY_MODE NODE_RANK MASTER_ADDR MASTER_PORT FORMAL_RUN_ID [trainer arguments...]" >&2
    exit 2
fi

python_bin=$1
topology_mode=$2
node_rank=$3
master_addr=$4
master_port=$5
formal_run_id=$6
shift 6
launcher_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
script_dir=$launcher_dir

if [[ ! -x "$python_bin" ]]; then
    printf 'Python executable is missing or not executable: %s\n' \
        "$python_bin" >&2
    exit 2
fi

case "$topology_mode" in
    official_w1_b64_reference)
        nnodes=1
        nproc_per_node=1
        local_batch_size=64
        learning_rate=0.00005
        precision=fp32
        ;;
    official_objective_w8_l8_g64_ddp_adaptation)
        nnodes=1
        nproc_per_node=8
        local_batch_size=8
        learning_rate=0.00005
        precision=bf16
        ;;
    official_objective_w16_l4_g64_ddp_adaptation)
        nnodes=2
        nproc_per_node=8
        local_batch_size=4
        learning_rate=0.00005
        precision=bf16
        ;;
    validation_gated_w8_l64_g512_empirical_acceleration)
        nnodes=1
        nproc_per_node=8
        local_batch_size=64
        learning_rate=0.00003
        precision=bf16
        ;;
    validation_gated_w16_l32_g512_empirical_acceleration)
        nnodes=2
        nproc_per_node=8
        local_batch_size=32
        learning_rate=0.00003
        precision=bf16
        ;;
    validation_gated_w8_l128_g1024_empirical_acceleration)
        nnodes=1
        nproc_per_node=8
        local_batch_size=128
        learning_rate=0.00003
        precision=bf16
        ;;
    validation_gated_w8_l256_g2048_empirical_acceleration)
        nnodes=1
        nproc_per_node=8
        local_batch_size=256
        learning_rate=0.00003
        precision=bf16
        ;;
    validation_gated_w16_l64_g1024_empirical_acceleration)
        nnodes=2
        nproc_per_node=8
        local_batch_size=64
        learning_rate=0.00003
        precision=bf16
        ;;
    validation_gated_w16_l64_g1024_lr6e5_empirical_acceleration)
        nnodes=2
        nproc_per_node=8
        local_batch_size=64
        learning_rate=0.00006
        precision=bf16
        ;;
    *)
        printf 'unknown immutable Base topology mode: %s\n' \
            "$topology_mode" >&2
        exit 2
        ;;
esac

case "$(hostname)" in
    iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0)
        host_slot=0
        ;;
    iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0)
        host_slot=1
        ;;
    *)
        printf 'host is not in the exact formal Base inventory\n' >&2
        exit 2
        ;;
esac
case "$node_rank" in
    0|1) ;;
    *)
        printf 'NODE_RANK must be exactly 0 or 1\n' >&2
        exit 2
        ;;
esac
if [[ "$node_rank" -ge "$nnodes" || \
      ( "$nnodes" -eq 2 && "$node_rank" -ne "$host_slot" ) || \
      ! "$master_addr" =~ ^[A-Za-z0-9.-]+$ || \
      ! "$master_port" =~ ^[0-9]+$ || \
      "$master_port" -lt 1024 || "$master_port" -gt 65535 || \
      ! "$formal_run_id" =~ ^[A-Za-z0-9._-]{8,128}$ ]]; then
    printf 'invalid formal Base node/host/static-rendezvous identity\n' >&2
    exit 2
fi

# shellcheck source=guarded_runner_contract.sh
. "$launcher_dir/guarded_runner_contract.sh"
semtalk_require_exact_guarded_runner_all_gpus

# PYTHONHASHSEED is read only at interpreter startup.  Bind it here, before
# torchrun creates the formal rank interpreters; the trainer independently
# verifies the value and fails closed if this launcher was bypassed.
export PYTHONHASHSEED=43
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec "$python_bin" -m torch.distributed.run \
    --nnodes="$nnodes" \
    --nproc_per_node="$nproc_per_node" \
    --node_rank="$node_rank" \
    --master_addr="$master_addr" \
    --master_port="$master_port" \
    "$script_dir/train_base_official_adapt_long.py" \
    --formal-node-rank "$node_rank" \
    --formal-host-slot "$host_slot" \
    --formal-master-addr "$master_addr" \
    --formal-master-port "$master_port" \
    --formal-run-id "$formal_run_id" \
    "$@" \
    --topology-mode "$topology_mode" \
    --local-batch-size "$local_batch_size" \
    --learning-rate "$learning_rate" \
    --precision "$precision"
