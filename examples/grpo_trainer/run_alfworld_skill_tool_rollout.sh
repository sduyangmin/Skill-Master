#!/bin/bash

# Anonymous release note:
# - Set MODEL_PATH to an SFT checkpoint before running.
# - If utility reward is enabled, provide ALFWORLD_PROBE_GAMEFILES_PATH
#   or place data/alfworld_game_json/alfworld_train_gamefiles.json in the repo.
# - Optional overrides can be supplied either via environment variables below
#   or via trailing Hydra arguments after the script invocation.

if [[ "${DEBUG_SCRIPT:-0}" == "1" ]]; then
    set -x
fi

ENGINE=${1:-vllm}
if [[ "$#" -gt 0 ]]; then
    shift
fi

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-warning}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export WANDB_NAME=${WANDB_NAME:-alfworld_grpo_turnlevel_skill_tool}
export SKILLRL_TMPDIR=${SKILLRL_TMPDIR:-$HOME/tmp/skillrl_tmp}
mkdir -p "$SKILLRL_TMPDIR"

# Required paths
model_path=${MODEL_PATH:?Please set MODEL_PATH}
base_skill_bank_path=${BASE_SKILL_BANK_PATH:-memory_data/alfworld/claude_style_skills.json}
skill_bank_path=${SKILL_BANK_PATH:-memory_data/alfworld/self_managed_skills.json}
probe_gamefiles_path=${ALFWORLD_PROBE_GAMEFILES_PATH:-data/alfworld_game_json/alfworld_train_gamefiles.json}
validation_dump_dir=${VALIDATION_DUMP_DIR:-debug/validation_rollouts}
skill_tool_config_path=${SKILL_TOOL_CONFIG_PATH:-examples/sglang_multiturn/config/tool_config/skill_bank_tool_config.yaml}

# Core rollout settings
num_gpus_per_node=${NUM_GPUS_PER_NODE:-8}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-1}
train_data_size=${TRAIN_DATA_SIZE:-8}
val_data_size=${VAL_DATA_SIZE:-128}
val_batch_size=${VAL_BATCH_SIZE:-16}
group_size=${GROUP_SIZE:-8}
history_length=${HISTORY_LENGTH:-12}

# Sequence limits
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
trajectory_response_length=${TRAJECTORY_RESPONSE_LENGTH:-16384}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-16384}
max_num_seqs=${MAX_NUM_SEQS:-64}

# Trainer schedule
experiment_name=${EXPERIMENT_NAME:-alfworld_grpo_turnlevel_skill_tool}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}
total_epochs=${TOTAL_EPOCHS:-300}

if [[ ! -f "$skill_bank_path" ]]; then
    mkdir -p "$(dirname "$skill_bank_path")"
    cp "$base_skill_bank_path" "$skill_bank_path"
fi

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size"

if [[ -n "${FIXED_EVAL_GAMEFILES:-}" ]]; then
    python3 scripts/attach_alfworld_eval_gamefiles.py \
        --input "$HOME/data/verl-agent/text/test.parquet" \
        --gamefiles "$FIXED_EVAL_GAMEFILES" \
        --output "$HOME/data/verl-agent/text/test.parquet"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$skill_tool_config_path \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ++env.history_length=$history_length \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=$skill_bank_path \
    +env.skills_only_memory.top_k=13 \
    +env.skills_only_memory.retrieve_on_init=True \
    +env.skills_only_memory.enable_dynamic_update=False \
    +env.skill_tool_rollout.enable=True \
    +env.skill_tool_rollout.apply_on_validation=True \
    +env.skill_tool_rollout.dump_validation_rollouts=True \
    +env.skill_tool_rollout.autosave=True \
    +env.skill_tool_rollout.skill_bank_path=$skill_bank_path \
    +env.skill_tool_rollout.trace_max_steps=12 \
    +env.skill_tool_rollout.trajectory_level=False \
    +env.skill_tool_rollout.validation_dump_dir=$validation_dump_dir \
    +env.skill_tool_rollout.mutate_on_train=False \
    +env.skill_tool_rollout.mutate_on_validation=True \
    +env.skill_tool_rollout.skill_mgmt_advantage_weight=0.2 \
    +env.skill_tool_rollout.trajectory_response_length=$trajectory_response_length \
    +env.skill_tool_rollout.trajectory_obs_max_chars=6000 \
    +env.skill_tool_rollout.utility_reward.enable=True \
    +env.skill_tool_rollout.utility_reward.apply_on_validation=False \
    +env.skill_tool_rollout.utility_reward.same_probe_k=4 \
    +env.skill_tool_rollout.utility_reward.alfworld_probe_gamefiles_path=$probe_gamefiles_path \
    +env.skill_tool_rollout.utility_reward.same_delta_win_loss_gamma=0.2 \
    +env.skill_tool_rollout.parse_error_penalty=-2.0 \
    +env.skill_tool_rollout.invalid_arguments_penalty=-1.0 \
    +env.skill_tool_rollout.unknown_tool_penalty=-1.5 \
    +env.skill_tool_rollout.valid_format_bonus=0.1 \
    +env.skill_tool_rollout.duplicate_penalty=-0.5 \
    +env.skill_tool_rollout.action_like_tool_penalty=-3.0 \
    +env.skill_tool_rollout.missing_think_penalty=-0.5 \
    +env.skill_tool_rollout.missing_tool_call_penalty=-0.5 \
    +env.skill_tool_rollout.placeholder_penalty=-1.5 \
    +env.skill_tool_rollout.action_response_clip_penalty=-2.0 \
    +env.skill_tool_rollout.action_missing_action_penalty=-2.0 \
    +env.skill_tool_rollout.action_missing_think_close_penalty=-1.0 \
    +env.skill_tool_rollout.action_empty_response_penalty=-2.0 \
    +env.skill_tool_rollout.action_repetition_penalty=-0.5 \
    +env.skill_tool_rollout.action_repetition_threshold=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$num_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=False \
    "$@"
