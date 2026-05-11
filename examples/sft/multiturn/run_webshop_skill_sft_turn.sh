#!/bin/bash
# Turn-level SFT for skill-integrated WebShop tool-use.
# Anonymous release note:
# - Provide local model checkpoints via positional args or DEFAULT_MODEL_PATH / MODEL_PATH.
# - Provide local dataset paths via TRAIN_DATA / VAL_DATA if you do not mirror the repo layout.
set -x

DEFAULT_NPROC_PER_NODE=${DEFAULT_NPROC_PER_NODE:-8}
DEFAULT_MODEL_PATH=${DEFAULT_MODEL_PATH:-${MODEL_PATH:-}}
DEFAULT_SAVE_PATH=${DEFAULT_SAVE_PATH:-checkpoints/webshop_skill_tool_sft_turn_qwen25_7b}

nproc_per_node=${1:-$DEFAULT_NPROC_PER_NODE}
model_path=${2:-$DEFAULT_MODEL_PATH}
save_path=${3:-$DEFAULT_SAVE_PATH}

if [[ -z "$model_path" ]]; then
    echo "Please provide model path as arg2 or set DEFAULT_MODEL_PATH / MODEL_PATH" >&2
    exit 1
fi

if [ "$#" -ge 3 ]; then
    shift 3
elif [ "$#" -eq 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi

TRAIN_DATA=${TRAIN_DATA:-data/skillrl_skill_tool_sft/webshop/train_turn.parquet}
VAL_DATA=${VAL_DATA:-$TRAIN_DATA}

DATA_MAX_LENGTH=${DATA_MAX_LENGTH:-6144}
DATA_TRUNCATION=${DATA_TRUNCATION:-right}
MICRO_BSZ=${MICRO_BSZ:-1}
TRAIN_BSZ=${TRAIN_BSZ:-16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
PROJECT_NAME=${PROJECT_NAME:-webshop-skill-tool-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-webshop-skill-sft-turn-qwen25-7b}
LOGGER_BACKENDS=${LOGGER_BACKENDS:-"['console','wandb']"}

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.max_length=$DATA_MAX_LENGTH \
    data.truncation=$DATA_TRUNCATION \
    data.micro_batch_size_per_gpu=$MICRO_BSZ \
    data.train_batch_size=$TRAIN_BSZ \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=$LOGGER_BACKENDS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    "$@"
