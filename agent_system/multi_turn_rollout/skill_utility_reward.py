from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout import skill_management
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from agent_system.skill_utility import ALFWorldProbeSelector
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.tools.skill_bank_tools import SkillBankStore


def _select_utility_candidate_indices(
    *,
    cfg,
    responses: list[str],
    tool_results: list[dict[str, Any]],
    mutate_tools: set[str],
    valid_status: set[str],
) -> set[int]:
    eligible = []
    for item, result in enumerate(tool_results):
        tool = str(result.get("tool") or "")
        status = str(result.get("status") or "")
        if tool in mutate_tools and status in valid_status:
            eligible.append(item)

    max_candidates = int(cfg.get("max_utility_candidates_per_batch", 0) or 0)
    if max_candidates <= 0 or len(eligible) <= max_candidates:
        return set(eligible)

    seed = int(cfg.get("seed", 0))
    hasher = hashlib.sha1()
    for item in eligible:
        hasher.update(str(item).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tool_results[item].get("tool") or "").encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tool_results[item].get("status") or "").encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(responses[item] or "").encode("utf-8"))
        hasher.update(b"\0")
    batch_seed = int.from_bytes(hasher.digest()[:8], "big") % (2**32)
    rng = np.random.RandomState(seed ^ batch_seed)
    selected = rng.choice(eligible, size=max_candidates, replace=False).tolist()
    return {int(x) for x in selected}


def compute_skill_downstream_utility_rewards(
    collector,
    actor_rollout_wg,
    envs: EnvironmentManagerBase,
    responses: list[str],
    tool_results: list[dict[str, Any]],
    is_train: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not collector._skill_utility_enabled(is_train=is_train):
        rewards = np.zeros(len(tool_results), dtype=np.float32)
        return rewards, [{} for _ in tool_results]
    if skill_management.is_alfworld_env(collector.config):
        return compute_alfworld_skill_downstream_utility_rewards(
            collector=collector,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            responses=responses,
            tool_results=tool_results,
        )
    if skill_management.is_webshop_env(collector.config):
        return compute_webshop_skill_downstream_utility_rewards(
            collector=collector,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            responses=responses,
            tool_results=tool_results,
        )
    if skill_management.is_search_env(collector.config):
        return compute_searchqa_skill_downstream_utility_rewards(
            collector=collector,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            responses=responses,
            tool_results=tool_results,
        )
    rewards = np.zeros(len(tool_results), dtype=np.float32)
    return rewards, [{} for _ in tool_results]


def compute_webshop_skill_downstream_utility_rewards(
    collector,
    actor_rollout_wg,
    envs: EnvironmentManagerBase,
    responses: list[str],
    tool_results: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rewards = np.zeros(len(tool_results), dtype=np.float32)
    probe_infos: list[dict[str, Any]] = []

    mutate_tools = {"propose_skill", "update_skill"}
    valid_status = {"added", "updated"}
    cfg = collector._skill_utility_cfg()
    selected_indices = _select_utility_candidate_indices(
        cfg=cfg,
        responses=responses,
        tool_results=tool_results,
        mutate_tools=mutate_tools,
        valid_status=valid_status,
    )
    for item, result in enumerate(tool_results):
        tool = str(result.get("tool") or "")
        status = str(result.get("status") or "")
        if tool not in mutate_tools or status not in valid_status:
            probe_infos.append({})
            continue
        if item not in selected_indices:
            probe_infos.append(
                {
                    "utility_skipped_due_to_batch_cap": True,
                    "max_utility_candidates_per_batch": int(cfg.get("max_utility_candidates_per_batch", 0) or 0),
                }
            )
            continue

        probe_info = select_webshop_skill_utility_probes(collector, envs, item)
        if not probe_info.get("same"):
            probe_infos.append(probe_info)
            continue

        base_skill_bank_path = collector._get_skill_bank_path()
        temp_skill_bank_path, bank_stats = apply_skill_mutation_to_bank_copy(
            collector,
            base_skill_bank_path=base_skill_bank_path,
            response=responses[item],
            tool_result=result,
            envs=envs,
            item=item,
        )
        if temp_skill_bank_path is None:
            probe_infos.append(probe_info)
            continue

        try:
            before_eval = evaluate_webshop_probe_batch(
                collector,
                actor_rollout_wg=actor_rollout_wg,
                skill_bank_path=base_skill_bank_path,
                probe_info=probe_info,
            )
            after_eval = evaluate_webshop_probe_batch(
                collector,
                actor_rollout_wg=actor_rollout_wg,
                skill_bank_path=temp_skill_bank_path,
                probe_info=probe_info,
            )
        finally:
            with contextlib.suppress(Exception):
                os.remove(temp_skill_bank_path)

        same_scores_before = [float(x) for x in before_eval.get("same_scores", [])]
        same_scores_after = [float(x) for x in after_eval.get("same_scores", [])]
        same_deltas = [a - b for a, b in zip(same_scores_after, same_scores_before)]
        mean_delta = float(np.mean(same_deltas)) if same_deltas else 0.0
        std_delta = float(np.std(same_deltas)) if same_deltas else 0.0

        win_loss_gamma = float(cfg.get("same_delta_win_loss_gamma", 0.3))
        bank_size_penalty_coef = float(cfg.get("bank_size_penalty_coef", 0.0))
        size_delta = float(bank_stats.get("after_bank_stats", {}).get("total", 0)) - float(
            bank_stats.get("before_bank_stats", {}).get("total", 0)
        )

        win_count = int(sum(delta > 0 for delta in same_deltas))
        lose_count = int(sum(delta < 0 for delta in same_deltas))
        net_win_ratio = float((win_count - lose_count) / len(same_deltas)) if same_deltas else 0.0
        same_reward = max(0.0, mean_delta + win_loss_gamma * net_win_ratio)
        rewards[item] = (
            same_reward
            - bank_size_penalty_coef * max(0.0, size_delta)
        )
        probe_infos.append(
            {
                **probe_info,
                **bank_stats,
                "before_eval": before_eval,
                "after_eval": after_eval,
                "same_deltas": same_deltas,
                "same_mean_delta": mean_delta,
                "same_std_delta": std_delta,
                "same_win_count": win_count,
                "same_lose_count": lose_count,
                "same_net_win_ratio": net_win_ratio,
                "same_reward": same_reward,
                "utility_reward": float(rewards[item]),
            }
        )

    return rewards, probe_infos


def compute_alfworld_skill_downstream_utility_rewards(
    collector,
    actor_rollout_wg,
    envs: EnvironmentManagerBase,
    responses: list[str],
    tool_results: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rewards = np.zeros(len(tool_results), dtype=np.float32)
    probe_infos: list[dict[str, Any]] = []

    mutate_tools = {"propose_skill", "update_skill"}
    valid_status = {"added", "updated"}
    cfg = collector._skill_utility_cfg()
    selected_indices = _select_utility_candidate_indices(
        cfg=cfg,
        responses=responses,
        tool_results=tool_results,
        mutate_tools=mutate_tools,
        valid_status=valid_status,
    )
    for item, result in enumerate(tool_results):
        tool = str(result.get("tool") or "")
        status = str(result.get("status") or "")
        if tool not in mutate_tools or status not in valid_status:
            probe_infos.append({})
            continue
        if item not in selected_indices:
            probe_infos.append(
                {
                    "utility_skipped_due_to_batch_cap": True,
                    "max_utility_candidates_per_batch": int(cfg.get("max_utility_candidates_per_batch", 0) or 0),
                }
            )
            continue
        probe_info = select_alfworld_skill_utility_probes(collector, envs, item)
        if not probe_info.get("same") and not probe_info.get("different"):
            probe_infos.append(probe_info)
            continue

        base_skill_bank_path = collector._get_skill_bank_path()
        temp_skill_bank_path, bank_stats = apply_skill_mutation_to_bank_copy(
            collector,
            base_skill_bank_path=base_skill_bank_path,
            response=responses[item],
            tool_result=result,
            envs=envs,
            item=item,
        )
        if temp_skill_bank_path is None:
            probe_infos.append(probe_info)
            continue

        try:
            before_eval = evaluate_alfworld_probe_batch(
                collector,
                actor_rollout_wg=actor_rollout_wg,
                skill_bank_path=base_skill_bank_path,
                probe_info=probe_info,
            )
            after_eval = evaluate_alfworld_probe_batch(
                collector,
                actor_rollout_wg=actor_rollout_wg,
                skill_bank_path=temp_skill_bank_path,
                probe_info=probe_info,
            )
        finally:
            with contextlib.suppress(Exception):
                os.remove(temp_skill_bank_path)

        same_scores_before = [float(x) for x in before_eval.get("same_scores", [])]
        same_scores_after = [float(x) for x in after_eval.get("same_scores", [])]
        same_deltas = [a - b for a, b in zip(same_scores_after, same_scores_before)]
        mean_delta = float(np.mean(same_deltas)) if same_deltas else 0.0
        std_delta = float(np.std(same_deltas)) if same_deltas else 0.0

        win_loss_gamma = float(cfg.get("same_delta_win_loss_gamma", 0.3))
        bank_size_penalty_coef = float(cfg.get("bank_size_penalty_coef", 0.0))
        size_delta = float(bank_stats.get("after_bank_stats", {}).get("total", 0)) - float(
            bank_stats.get("before_bank_stats", {}).get("total", 0)
        )

        win_count = int(sum(delta > 0 for delta in same_deltas))
        lose_count = int(sum(delta < 0 for delta in same_deltas))
        net_win_ratio = float((win_count - lose_count) / len(same_deltas)) if same_deltas else 0.0
        same_reward = max(0.0, mean_delta + win_loss_gamma * net_win_ratio)
        rewards[item] = (
            same_reward
            - bank_size_penalty_coef * max(0.0, size_delta)
        )
        probe_infos.append(
            {
                **probe_info,
                **bank_stats,
                "before_eval": before_eval,
                "after_eval": after_eval,
                "same_deltas": same_deltas,
                "same_mean_delta": mean_delta,
                "same_std_delta": std_delta,
                "same_win_count": win_count,
                "same_lose_count": lose_count,
                "same_net_win_ratio": net_win_ratio,
                "same_reward": same_reward,
                "utility_reward": float(rewards[item]),
            }
        )

    return rewards, probe_infos


# ── SearchQA probe evaluation ─────────────────────────────────────────────


_SEARCHQA_PROBE_CACHE: dict[str, Any] = {}


def _get_searchqa_probe_data(probe_index_dir: str) -> dict[str, Any]:
    """Load (and cache) the SearchQA probe catalog, index, and embeddings."""
    import pandas as pd

    cache_key = os.path.abspath(str(probe_index_dir))
    if cache_key in _SEARCHQA_PROBE_CACHE:
        return _SEARCHQA_PROBE_CACHE[cache_key]

    catalog = pd.read_parquet(os.path.join(probe_index_dir, "catalog.parquet"))
    probe_index = pd.read_parquet(os.path.join(probe_index_dir, "probe_index.parquet"))
    embeddings = np.load(os.path.join(probe_index_dir, "embeddings.npy"))
    with open(os.path.join(probe_index_dir, "metadata.json")) as fh:
        metadata = json.load(fh)

    # Build question → row index mapping
    question_to_row: dict[str, int] = {}
    for _, row in catalog.iterrows():
        question_to_row[str(row["question"]).strip()] = int(row["row_index"])

    data = {
        "catalog": catalog,
        "probe_index": probe_index,
        "embeddings": embeddings,
        "metadata": metadata,
        "question_to_row": question_to_row,
    }
    _SEARCHQA_PROBE_CACHE[cache_key] = data
    return data


def select_searchqa_skill_utility_probes(
    collector,
    envs: EnvironmentManagerBase,
    item: int,
) -> dict[str, Any]:
    """Select probe tasks similar to the task at *item* using pre-computed embeddings."""
    tasks = getattr(envs, "tasks", [])
    task = str(tasks[item] if item < len(tasks) else "")
    if not task:
        return {"same": [], "different": []}

    cfg = collector._skill_utility_cfg()
    probe_index_dir = cfg.get("searchqa_probe_index_dir", "")
    if not probe_index_dir:
        return {"same": [], "different": []}

    data = _get_searchqa_probe_data(probe_index_dir)
    row_idx = data["question_to_row"].get(task.strip())
    if row_idx is None:
        return {"same": [], "different": []}

    idx_row = data["probe_index"][data["probe_index"]["row_index"] == row_idx]
    if idx_row.empty:
        return {"same": [], "different": []}

    same_k = int(cfg.get("same_probe_k", 4))

    same_indices = list(idx_row.iloc[0]["same_candidate_indices"][:same_k])
    catalog = data["catalog"]
    same_tasks: list[dict[str, Any]] = []
    for si in same_indices:
        si_row = catalog[catalog["row_index"] == int(si)]
        if si_row.empty:
            continue
        si_row = si_row.iloc[0]
        same_tasks.append({
            "task_payload": {
                "question": str(si_row["question"]),
                "data_source": str(si_row["data_source"]),
                "query_family": str(si_row["query_family"]),
                "query_subtype": str(si_row["query_subtype"]),
            }
        })

    return {
        "same": same_tasks,
        "different": [],
        "query_family": str(idx_row.iloc[0].get("query_family", "")) if not idx_row.empty else "",
        "task": task,
    }


def get_searchqa_probe_env_manager(
    collector,
    probe_tasks: list[dict[str, Any]],
) -> EnvironmentManagerBase:
    """Build a SearchEnvironmentManager pre-loaded with the given probe questions."""
    from agent_system.environments.env_package.search.envs import build_search_envs
    from agent_system.environments.env_package.search import search_projection
    from agent_system.environments.env_manager import SearchEnvironmentManager

    batch_size = len(probe_tasks)
    seed_offset = int(collector._skill_utility_cfg().get("probe_seed_offset", 50000))
    seed = int(collector.config.env.seed) + seed_offset + batch_size
    resources_per_worker = OmegaConf.to_container(
        collector.config.env.resources_per_worker, resolve=True
    )

    probe_envs = build_search_envs(
        seed=seed,
        env_num=batch_size,
        group_n=1,
        is_train=False,
        env_config=collector.config.env,
    )
    probe_manager = SearchEnvironmentManager(
        probe_envs, partial(search_projection), collector.config
    )

    # Override tasks with probe questions
    probe_manager.tasks = [
        t["task_payload"]["question"] for t in probe_tasks
    ]
    # Ensure retrieval memory is set up for probes
    if collector.config.env.get("use_skills_only_memory", False):
        from agent_system.memory import SkillsOnlyMemory
        som_cfg = collector.config.env.skills_only_memory
        probe_manager.retrieval_memory = SkillsOnlyMemory(
            skills_json_path=som_cfg.skills_json_path,
            retrieval_mode=som_cfg.get("retrieval_mode", "template"),
            embedding_model_path=som_cfg.get("embedding_model_path", None),
            task_specific_top_k=som_cfg.get("task_specific_top_k", None),
        )
        probe_manager.retrieved_memories = [
            probe_manager.retrieval_memory.retrieve(
                task_description=t,
                top_k=som_cfg.get("top_k", 10),
            )
            for t in probe_manager.tasks
        ]

    return probe_manager


def build_searchqa_probe_gen_batch(
    collector,
    probe_tasks: list[dict[str, Any]],
) -> DataProto:
    batch_size = len(probe_tasks)
    tensors = {
        "input_ids": torch.zeros((batch_size, 1), dtype=torch.long),
    }
    non_tensors = {
        "raw_prompt": np.array([[] for _ in range(batch_size)], dtype=object),
        "data_source": np.array(["searchqa_probe"] * batch_size, dtype=object),
        "env_kwargs": np.array(
            [
                {
                    "question": str(t["task_payload"]["question"]),
                    "ground_truth": "",
                    "data_source": str(t["task_payload"].get("data_source", "")),
                }
                for t in probe_tasks
            ],
            dtype=object,
        ),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={})


def evaluate_searchqa_probe_batch(
    collector,
    *,
    actor_rollout_wg,
    skill_bank_path: str,
    probe_info: dict[str, Any],
) -> dict[str, Any]:
    cfg = collector._skill_utility_cfg()
    max_steps = int(cfg.get("probe_max_steps", collector.config.env.max_steps))
    same_tasks = list(probe_info.get("same", []))

    if not same_tasks:
        return {
            "same_metrics": {},
            "different_metrics": {},
            "same_score": 0.0,
            "different_score": 0.0,
            "same_scores": [],
            "different_scores": [],
        }

    same_envs = get_searchqa_probe_env_manager(collector, same_tasks)
    with temporary_alfworld_skill_bank(collector, same_envs, skill_bank_path) as enabled:
        if enabled:
            same_batch = build_searchqa_probe_gen_batch(collector, same_tasks)
            same_metrics = run_probe_action_rollout(
                collector,
                gen_batch=same_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=same_envs,
                max_steps=max_steps,
            )

    return {
        "same_metrics": same_metrics if same_tasks else {},
        "different_metrics": {},
        "same_score": probe_score_from_metrics(collector, same_metrics) if same_metrics else 0.0,
        "different_score": 0.0,
        "same_scores": list(same_metrics.get("per_probe_scores", [])) if same_metrics else [],
        "different_scores": [],
    }


def compute_searchqa_skill_downstream_utility_rewards(
    collector,
    actor_rollout_wg,
    envs: EnvironmentManagerBase,
    responses: list[str],
    tool_results: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rewards = np.zeros(len(tool_results), dtype=np.float32)
    probe_infos: list[dict[str, Any]] = []

    mutate_tools = {"propose_skill", "update_skill"}
    valid_status = {"added", "updated"}
    cfg = collector._skill_utility_cfg()
    selected_indices = _select_utility_candidate_indices(
        cfg=cfg,
        responses=responses,
        tool_results=tool_results,
        mutate_tools=mutate_tools,
        valid_status=valid_status,
    )
    for item, result in enumerate(tool_results):
        tool = str(result.get("tool") or "")
        status = str(result.get("status") or "")
        if tool not in mutate_tools or status not in valid_status:
            probe_infos.append({})
            continue
        if item not in selected_indices:
            probe_infos.append(
                {
                    "utility_skipped_due_to_batch_cap": True,
                    "max_utility_candidates_per_batch": int(cfg.get("max_utility_candidates_per_batch", 0) or 0),
                }
            )
            continue

        probe_info = select_searchqa_skill_utility_probes(collector, envs, item)
        if not probe_info.get("same"):
            probe_infos.append(probe_info)
            continue

        base_skill_bank_path = collector._get_skill_bank_path()
        temp_skill_bank_path, bank_stats = apply_skill_mutation_to_bank_copy(
            collector,
            base_skill_bank_path=base_skill_bank_path,
            response=responses[item],
            tool_result=result,
            envs=envs,
            item=item,
        )
        if temp_skill_bank_path is None:
            probe_infos.append(probe_info)
            continue

        try:
            before_eval = evaluate_searchqa_probe_batch(
                collector,
                actor_rollout_wg=actor_rollout_wg,
                skill_bank_path=base_skill_bank_path,
                probe_info=probe_info,
            )
            after_eval = evaluate_searchqa_probe_batch(
                collector,
                actor_rollout_wg=actor_rollout_wg,
                skill_bank_path=temp_skill_bank_path,
                probe_info=probe_info,
            )
        finally:
            with contextlib.suppress(Exception):
                os.remove(temp_skill_bank_path)

        same_scores_before = [float(x) for x in before_eval.get("same_scores", [])]
        same_scores_after = [float(x) for x in after_eval.get("same_scores", [])]
        same_deltas = [a - b for a, b in zip(same_scores_after, same_scores_before)]
        mean_delta = float(np.mean(same_deltas)) if same_deltas else 0.0
        std_delta = float(np.std(same_deltas)) if same_deltas else 0.0

        win_loss_gamma = float(cfg.get("same_delta_win_loss_gamma", 0.3))
        bank_size_penalty_coef = float(cfg.get("bank_size_penalty_coef", 0.0))
        size_delta = float(bank_stats.get("after_bank_stats", {}).get("total", 0)) - float(
            bank_stats.get("before_bank_stats", {}).get("total", 0)
        )

        win_count = int(sum(delta > 0 for delta in same_deltas))
        lose_count = int(sum(delta < 0 for delta in same_deltas))
        net_win_ratio = float((win_count - lose_count) / len(same_deltas)) if same_deltas else 0.0
        same_reward = max(0.0, mean_delta + win_loss_gamma * net_win_ratio)
        rewards[item] = (
            same_reward
            - bank_size_penalty_coef * max(0.0, size_delta)
        )
        probe_infos.append(
            {
                **probe_info,
                **bank_stats,
                "before_eval": before_eval,
                "after_eval": after_eval,
                "same_deltas": same_deltas,
                "same_mean_delta": mean_delta,
                "same_std_delta": std_delta,
                "same_win_count": win_count,
                "same_lose_count": lose_count,
                "same_net_win_ratio": net_win_ratio,
                "same_reward": same_reward,
                "utility_reward": float(rewards[item]),
            }
        )

    return rewards, probe_infos


def get_alfworld_probe_selector(collector) -> ALFWorldProbeSelector | None:
    cfg = collector._skill_utility_cfg()
    gamefiles_path = cfg.get("alfworld_probe_gamefiles_path", None)
    if not gamefiles_path:
        return None

    same_k = int(cfg.get("same_probe_k", 4))
    different_k = int(cfg.get("different_probe_k", 2))
    seed = int(cfg.get("seed", 0))
    cache_key = (str(gamefiles_path), same_k, different_k, seed)
    selector = collector._alfworld_probe_selector
    if selector is None or getattr(selector, "_cache_key", None) != cache_key:
        selector = ALFWorldProbeSelector(
            gamefiles_path=str(gamefiles_path),
            same_k=same_k,
            different_k=different_k,
            seed=seed,
        )
        selector._cache_key = cache_key
        collector._alfworld_probe_selector = selector
    return selector


def _normalize_webshop_goal_text(goal: Any) -> str:
    if goal is None:
        return ""
    if isinstance(goal, str):
        return goal
    if isinstance(goal, dict):
        for key in ("instruction_text", "goal", "task", "text"):
            value = goal.get(key)
            if value:
                return str(value)
    return str(goal)


def get_webshop_probe_goal_catalog(collector, envs: EnvironmentManagerBase) -> list[tuple[int, str]]:
    raw_envs = getattr(envs, "envs", None)
    goal_indices = getattr(raw_envs, "goal_idxs", [])
    all_goals = getattr(raw_envs, "goals", [])
    catalog: list[tuple[int, str]] = []
    for idx in goal_indices:
        try:
            goal_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= goal_idx < len(all_goals):
            catalog.append((goal_idx, _normalize_webshop_goal_text(all_goals[goal_idx])))
    return catalog


def select_webshop_skill_utility_probes(
    collector,
    envs: EnvironmentManagerBase,
    item: int,
) -> dict[str, Any]:
    current_goal_idx = skill_management.get_webshop_goal_idx(envs, item)
    current_goal = skill_management.get_webshop_goal(envs, item)
    tasks = getattr(envs, "tasks", [])
    current_task = tasks[item] if item < len(tasks) else ""
    category = skill_management.infer_webshop_skill_category(current_task or str(current_goal or ""))

    catalog = get_webshop_probe_goal_catalog(collector, envs)
    same_candidates = []
    for goal_idx, goal_text in catalog:
        if current_goal_idx is not None and goal_idx == int(current_goal_idx):
            continue
        if skill_management.infer_webshop_skill_category(goal_text) != category:
            continue
        same_candidates.append(
            {
                "task_payload": {
                    "goal_idx": int(goal_idx),
                    "goal": goal_text,
                }
            }
        )

    cfg = collector._skill_utility_cfg()
    same_k = int(cfg.get("same_probe_k", 2))
    seed = int(cfg.get("seed", 0))
    rng = np.random.RandomState(seed + (int(current_goal_idx) if current_goal_idx is not None else item))
    if len(same_candidates) > same_k:
        selected_indices = rng.choice(len(same_candidates), size=same_k, replace=False).tolist()
        same_selected = [same_candidates[i] for i in selected_indices]
    else:
        same_selected = same_candidates

    return {
        "same": same_selected,
        "different": [],
        "category": category,
        "goal_idx": current_goal_idx,
        "goal": _normalize_webshop_goal_text(current_goal),
    }


def get_alfworld_probe_env_manager(collector, batch_size: int, split: str | None = None) -> EnvironmentManagerBase:
    normalized_split = str(split or "").strip().lower() or "eval"
    cache_key = ("alfworld", int(batch_size), normalized_split)
    if cache_key in collector._probe_env_managers:
        return collector._probe_env_managers[cache_key]

    from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager

    if collector.config.env.env_name == "alfworld/AlfredThorEnv":
        alf_config_path = os.path.join(
            os.path.dirname(__file__),
            "../environments/env_package/alfworld/configs/config_tw.yaml",
        )
    elif collector.config.env.env_name == "alfworld/AlfredTWEnv":
        alf_config_path = os.path.join(
            os.path.dirname(__file__),
            "../environments/env_package/alfworld/configs/config_tw.yaml",
        )
    else:
        raise ValueError(
            f"ALFWorld probe utility only supports ALFWorld envs, got {collector.config.env.env_name}"
        )
    alf_config_path = os.path.abspath(alf_config_path)

    resources_per_worker = OmegaConf.to_container(collector.config.env.resources_per_worker, resolve=True)
    is_train_split = normalized_split == "train"
    env_kwargs = {
        "eval_dataset": collector.config.env.alfworld.eval_dataset,
    }
    seed_offset = int(collector._skill_utility_cfg().get("probe_seed_offset", 50000))
    probe_envs = build_alfworld_envs(
        alf_config_path,
        collector.config.env.seed + seed_offset + int(batch_size),
        int(batch_size),
        1,
        is_train=is_train_split,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    projection_f = partial(alfworld_projection)
    probe_env_manager = AlfWorldEnvironmentManager(probe_envs, projection_f, collector.config)
    collector._probe_env_managers[cache_key] = probe_env_manager
    return probe_env_manager


def _probe_tasks_split(probe_tasks: list[dict[str, Any]]) -> str | None:
    if not probe_tasks:
        return None
    payload = probe_tasks[0].get("task_payload") or {}
    split = str(payload.get("split") or "").strip().lower()
    return split or None


def get_webshop_probe_env_manager(collector, batch_size: int) -> EnvironmentManagerBase:
    cache_key = ("webshop", int(batch_size))
    if cache_key in collector._probe_env_managers:
        return collector._probe_env_managers[cache_key]

    from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
    from agent_system.environments.env_manager import WebshopEnvironmentManager

    if collector.config.env.webshop.use_small:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "../environments/env_package/webshop/webshop/data/items_shuffle_1000.json",
        )
        attr_path = os.path.join(
            os.path.dirname(__file__),
            "../environments/env_package/webshop/webshop/data/items_ins_v2_1000.json",
        )
    else:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "../environments/env_package/webshop/webshop/data/items_shuffle.json",
        )
        attr_path = os.path.join(
            os.path.dirname(__file__),
            "../environments/env_package/webshop/webshop/data/items_ins_v2.json",
        )
    file_path = os.path.abspath(file_path)
    attr_path = os.path.abspath(attr_path)

    resources_per_worker = OmegaConf.to_container(collector.config.env.resources_per_worker, resolve=True)
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": collector.config.env.webshop.human_goals,
        "file_path": file_path,
        "attr_path": attr_path,
    }
    seed_offset = int(collector._skill_utility_cfg().get("probe_seed_offset", 50000))
    probe_envs = build_webshop_envs(
        seed=collector.config.env.seed + seed_offset + int(batch_size),
        env_num=int(batch_size),
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    probe_env_manager = WebshopEnvironmentManager(probe_envs, partial(webshop_projection), collector.config)
    collector._probe_env_managers[cache_key] = probe_env_manager
    return probe_env_manager


def select_alfworld_skill_utility_probes(
    collector,
    envs: EnvironmentManagerBase,
    item: int,
) -> dict[str, Any]:
    selector = get_alfworld_probe_selector(collector)
    gamefile = collector._get_alfworld_gamefile(envs, item)
    if selector is None or not gamefile:
        return {"same": [], "different": [], "family": None, "gamefile": gamefile}

    tasks = getattr(envs, "tasks", [])
    task = tasks[item] if item < len(tasks) else ""
    family = collector._infer_alfworld_success_family(task, gamefile=gamefile)
    probes = selector.select(gamefile=gamefile, family=family)
    payload = probes.to_jsonable()
    payload["family"] = family
    payload["gamefile"] = gamefile
    return payload


@contextmanager
def temporary_alfworld_skill_bank(
    collector,
    envs: EnvironmentManagerBase,
    skill_bank_path: str,
):
    original_retrieval_memory = getattr(envs, "retrieval_memory", None)
    original_retrieved_memories = getattr(envs, "retrieved_memories", None)
    original_tasks = getattr(envs, "tasks", None)
    original_gamefile = getattr(envs, "gamefile", None)
    original_pre_text_obs = getattr(envs, "pre_text_obs", None)

    if not collector.config.env.get("use_skills_only_memory", False):
        yield False
        return

    from agent_system.memory import SkillsOnlyMemory

    som_cfg = collector.config.env.skills_only_memory
    envs.retrieval_memory = SkillsOnlyMemory(
        skills_json_path=skill_bank_path,
        retrieval_mode=som_cfg.get("retrieval_mode", "template"),
        embedding_model_path=som_cfg.get("embedding_model_path", None),
        task_specific_top_k=som_cfg.get("task_specific_top_k", None),
    )
    envs.retrieved_memories = None
    try:
        yield True
    finally:
        envs.retrieval_memory = original_retrieval_memory
        envs.retrieved_memories = original_retrieved_memories
        if original_tasks is not None:
            envs.tasks = original_tasks
        if original_gamefile is not None:
            envs.gamefile = original_gamefile
        if original_pre_text_obs is not None:
            envs.pre_text_obs = original_pre_text_obs


def apply_skill_mutation_to_bank_copy(
    collector,
    *,
    base_skill_bank_path: str,
    response: str,
    tool_result: dict[str, Any],
    envs: EnvironmentManagerBase,
    item: int,
) -> tuple[str | None, dict[str, Any]]:
    name, arguments, error = collector._parse_skill_tool_call(response)
    if error is not None:
        return None, {}

    with open(base_skill_bank_path, "r", encoding="utf-8") as f:
        bank = json.load(f)

    store = SkillBankStore(skill_bank_path=base_skill_bank_path, autosave=False)
    bank = store._ensure_bank_structure(bank)
    before_stats = store.get_stats(bank)
    bank_after = copy.deepcopy(bank)

    if name == "propose_skill" and str(tool_result.get("status") or "") == "added":
        new_skill = copy.deepcopy(tool_result.get("skill") or {})
        category = str(arguments.get("category") or "").strip()
        if not new_skill or not category:
            return None, {}
        if category == "general":
            bank_after["general_skills"].append(new_skill)
        else:
            bank_after["task_specific_skills"].setdefault(category, []).append(new_skill)
    elif name == "update_skill" and str(tool_result.get("status") or "") == "updated":
        skill_id = str(arguments.get("skill_id") or "").strip()
        located = store.find_skill(bank_after, skill_id)
        if located is None:
            return None, {}
        _, _, skill = located
        new_skill = tool_result.get("new_skill") or {}
        skill["title"] = str(new_skill.get("title", skill.get("title", ""))).strip()
        skill["principle"] = str(new_skill.get("principle", skill.get("principle", ""))).strip()
        skill["when_to_apply"] = str(new_skill.get("when_to_apply", skill.get("when_to_apply", ""))).strip()
    else:
        return None, {}

    after_stats = store.get_stats(bank_after)
    tmp_root = os.environ.get("SKILLRL_TMPDIR") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    os.makedirs(tmp_root, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="skill_bank_probe_",
        dir=tmp_root,
        delete=False,
        encoding="utf-8",
    ) as f:
        json.dump(bank_after, f, indent=2, ensure_ascii=False)
        temp_path = f.name

    return temp_path, {
        "before_bank_stats": before_stats,
        "after_bank_stats": after_stats,
    }


def build_probe_gen_batch(collector, probe_tasks: list[dict[str, Any]]) -> DataProto:
    batch_size = len(probe_tasks)
    tensors = {
        "input_ids": torch.zeros((batch_size, 1), dtype=torch.long),
    }
    non_tensors = {
        "raw_prompt": np.array([[] for _ in range(batch_size)], dtype=object),
        "data_source": np.array(["alfworld_probe"] * batch_size, dtype=object),
        "env_kwargs": np.array(
            [
                {
                    "gamefile": str(task["task_payload"]["gamefile"]),
                    "split": str(task["task_payload"].get("split") or "").strip(),
                }
                for task in probe_tasks
            ],
            dtype=object,
        ),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={})


def build_webshop_probe_gen_batch(collector, probe_tasks: list[dict[str, Any]]) -> DataProto:
    batch_size = len(probe_tasks)
    tensors = {
        "input_ids": torch.zeros((batch_size, 1), dtype=torch.long),
    }
    non_tensors = {
        "raw_prompt": np.array([[] for _ in range(batch_size)], dtype=object),
        "data_source": np.array(["webshop_probe"] * batch_size, dtype=object),
        "env_kwargs": np.array(
            [{"goal_idx": int(task["task_payload"]["goal_idx"])} for task in probe_tasks],
            dtype=object,
        ),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={})


def run_probe_action_rollout(
    collector,
    *,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs: EnvironmentManagerBase,
    max_steps: int,
) -> dict[str, float]:
    batch_size = len(gen_batch.batch)
    obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))

    is_done = np.zeros(batch_size, dtype=bool)
    total_batch_list = [[] for _ in range(batch_size)]
    total_infos = [[] for _ in range(batch_size)]
    episode_lengths = np.zeros(batch_size, dtype=np.float32)
    episode_rewards = np.zeros(batch_size, dtype=np.float32)
    valid_action_counts = np.zeros(batch_size, dtype=np.float32)
    action_counts = np.zeros(batch_size, dtype=np.float32)

    for _step in range(max_steps):
        active_masks = np.logical_not(is_done)
        batch = collector.preprocess_batch(gen_batch=gen_batch, obs=obs)
        batch.non_tensor_batch["prompt_text"] = np.array(obs.get("text", [""] * batch_size), dtype=object)

        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        batch_input = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        batch = batch.union(batch_output)

        raw_text_actions = collector.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        next_obs, rewards, dones, infos = envs.step(list(raw_text_actions))
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)
        if len(dones.shape) == 2:
            dones = dones.squeeze(1)

        prompt_length = batch.batch["prompts"].shape[-1]
        response_attention_masks = batch.batch["attention_mask"][:, prompt_length:]
        rewards, dones, infos, _, _ = collector._apply_rollout_guard(
            responses=raw_text_actions,
            response_token_counts=np.array(
                [int(response_attention_masks[i].sum().item()) for i in range(batch_size)],
                dtype=np.int32,
            ),
            rewards=rewards,
            dones=dones,
            infos=infos,
        )
        if "is_action_valid" in infos[0]:
            is_action_valid = np.array([info["is_action_valid"] for info in infos], dtype=bool)
        else:
            is_action_valid = np.ones(batch_size, dtype=bool)

        batch.non_tensor_batch["is_action_valid"] = is_action_valid
        batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)
        batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
        batch.non_tensor_batch["is_skill_management_turn"] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch["skill_reward_shaping"] = np.zeros(batch_size, dtype=object)
        batch.non_tensor_batch["is_skill_tool_valid"] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch["skill_tool_result"] = np.array([""] * batch_size, dtype=object)
        batch.non_tensor_batch["skill_tool_name"] = np.array([""] * batch_size, dtype=object)
        batch.non_tensor_batch["skill_utility_reward"] = np.zeros(batch_size, dtype=object)
        batch.non_tensor_batch["skill_utility_probe_info"] = np.array([""] * batch_size, dtype=object)

        batch_list = to_list_of_dict(batch)
        valid_action_counts[active_masks] += is_action_valid.astype(np.float32)[active_masks]
        action_counts[active_masks] += 1.0
        episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
        episode_lengths[active_masks] += 1

        for i in range(batch_size):
            if not active_masks[i]:
                continue
            total_batch_list[i].append(batch_list[i])
            total_infos[i].append(infos[i])

        is_done = np.logical_or(is_done, dones)
        obs = next_obs
        if is_done.all():
            break

    success = envs.success_evaluator(
        total_infos=total_infos,
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )
    success_rates = np.asarray(
        success.get("success_rate", np.zeros(batch_size, dtype=np.float32)),
        dtype=np.float32,
    )
    valid_action_ratio = np.divide(
        valid_action_counts,
        np.maximum(action_counts, 1.0),
        out=np.zeros_like(valid_action_counts),
        where=np.maximum(action_counts, 1.0) > 0,
    )
    successful_mask = success_rates == 1.0
    if np.any(successful_mask):
        successful_step_efficiency = float(
            np.mean((float(max_steps) - episode_lengths[successful_mask]) / float(max_steps))
        )
    else:
        successful_step_efficiency = 0.0
    per_probe_scores = (
        success_rates
        + np.where(
            successful_mask,
            (float(max_steps) - episode_lengths) / float(max_steps),
            0.0,
        ).astype(np.float32)
    )
    return {
        "success_rate": float(np.mean(success_rates)),
        "successful_step_efficiency": successful_step_efficiency,
        "avg_reward": float(np.mean(episode_rewards)),
        "valid_action_ratio": float(np.mean(valid_action_ratio)),
        "per_probe_scores": [float(x) for x in per_probe_scores.tolist()],
        "per_probe_success_rate": [float(x) for x in success_rates.tolist()],
        "per_probe_episode_lengths": [float(x) for x in episode_lengths.tolist()],
    }


def probe_score_from_metrics(collector, metrics: dict[str, float]) -> float:
    return (
        float(metrics.get("success_rate", 0.0))
        + float(metrics.get("successful_step_efficiency", 0.0))
    )


def evaluate_alfworld_probe_batch(
    collector,
    *,
    actor_rollout_wg,
    skill_bank_path: str,
    probe_info: dict[str, Any],
) -> dict[str, Any]:
    cfg = collector._skill_utility_cfg()
    max_steps = int(cfg.get("probe_max_steps", collector.config.env.max_steps))
    same_tasks = list(probe_info.get("same", []))

    if not same_tasks:
        return {
            "same_metrics": {},
            "different_metrics": {},
            "same_score": 0.0,
            "different_score": 0.0,
            "same_scores": [],
            "different_scores": [],
        }

    same_metrics = {}
    if same_tasks:
        same_envs = get_alfworld_probe_env_manager(
            collector,
            len(same_tasks),
            split=_probe_tasks_split(same_tasks),
        )
        with temporary_alfworld_skill_bank(collector, same_envs, skill_bank_path) as enabled:
            if enabled:
                same_batch = build_probe_gen_batch(collector, same_tasks)
                same_metrics = run_probe_action_rollout(
                    collector,
                    gen_batch=same_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=same_envs,
                    max_steps=max_steps,
                )

    return {
        "same_metrics": same_metrics,
        "different_metrics": {},
        "same_score": probe_score_from_metrics(collector, same_metrics) if same_metrics else 0.0,
        "different_score": 0.0,
        "same_scores": list(same_metrics.get("per_probe_scores", [])) if same_metrics else [],
        "different_scores": [],
    }


def evaluate_webshop_probe_batch(
    collector,
    *,
    actor_rollout_wg,
    skill_bank_path: str,
    probe_info: dict[str, Any],
) -> dict[str, Any]:
    cfg = collector._skill_utility_cfg()
    max_steps = int(cfg.get("probe_max_steps", collector.config.env.max_steps))
    same_tasks = list(probe_info.get("same", []))

    if not same_tasks:
        return {
            "same_metrics": {},
            "different_metrics": {},
            "same_score": 0.0,
            "different_score": 0.0,
            "same_scores": [],
            "different_scores": [],
        }

    same_metrics = {}
    same_envs = get_webshop_probe_env_manager(collector, len(same_tasks))
    with temporary_alfworld_skill_bank(collector, same_envs, skill_bank_path) as enabled:
        if enabled:
            same_batch = build_webshop_probe_gen_batch(collector, same_tasks)
            same_metrics = run_probe_action_rollout(
                collector,
                gen_batch=same_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=same_envs,
                max_steps=max_steps,
            )

    return {
        "same_metrics": same_metrics,
        "different_metrics": {},
        "same_score": probe_score_from_metrics(collector, same_metrics) if same_metrics else 0.0,
        "different_score": 0.0,
        "same_scores": list(same_metrics.get("per_probe_scores", [])) if same_metrics else [],
        "different_scores": [],
    }
