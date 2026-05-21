#!/bin/bash

TRAIN_CONFIG="${TRAIN_CONFIG:-./configs/uika_base.yaml}"
TRAIN_RUNNER="${TRAIN_RUNNER:-train.uika}"
MAIN_PORT="${MAIN_PORT:-12346}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

NUM_PROCESSES=$((NUM_GPUS * NUM_MACHINES))

MULTI_NODE_ARGS=""
if [ -n "$NODE_RANK" ] && [ -n "$MASTER_ADDR" ]; then
  echo "Multi-node mode | Rank: ${NODE_RANK} | Master: ${MASTER_ADDR}:${MAIN_PORT}"
  MULTI_NODE_ARGS="--machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --same_network"
fi

echo "Launch: ${NUM_MACHINES} machine(s) x ${NUM_GPUS} GPU(s) = ${NUM_PROCESSES} processes | runner=${TRAIN_RUNNER} | config=${TRAIN_CONFIG} | mixed_precision=${MIXED_PRECISION}"

accelerate launch \
  --multi_gpu \
  --mixed_precision $MIXED_PRECISION \
  --num_machines $NUM_MACHINES \
  --num_processes $NUM_PROCESSES \
  --rdzv_backend static \
  $MULTI_NODE_ARGS \
  --main_process_port $MAIN_PORT \
  -m uika.launch $TRAIN_RUNNER --config $TRAIN_CONFIG
