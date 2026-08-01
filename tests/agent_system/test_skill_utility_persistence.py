import json
from types import SimpleNamespace

import numpy as np

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def _collector_with_utility_rewards(utility_rewards):
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    calls = []

    collector._skill_bank_mutation_enabled = lambda is_train: True
    collector._skill_utility_enabled = lambda is_train: True

    def execute(envs, response, item=None, mutate_bank=True):
        calls.append((response, mutate_bank))
        if response.startswith("propose"):
            return {
                "status": "added",
                "tool": "propose_skill",
                "dry_run": not mutate_bank,
                "bank_mutated": mutate_bank,
            }
        if response.startswith("update"):
            return {
                "status": "updated",
                "tool": "update_skill",
                "dry_run": not mutate_bank,
                "bank_mutated": mutate_bank,
            }
        return {
            "status": "kept",
            "tool": "keep_skill",
            "dry_run": not mutate_bank,
            "bank_mutated": False,
        }

    def score(actor_rollout_wg, envs, responses, tool_results, is_train):
        if collector._skill_utility_enabled(is_train):
            assert all(not mutate_bank for _, mutate_bank in calls)
        return np.asarray(utility_rewards, dtype=np.float32), [{} for _ in responses]

    collector._execute_skill_tool_call = execute
    collector._skill_downstream_utility_rewards = score
    return collector, calls


def test_only_positive_utility_skill_edits_are_persisted():
    collector, calls = _collector_with_utility_rewards([0.5, 0.0, -0.25, 0.0])

    results, rewards, _ = collector._execute_skill_tool_calls_with_utility_gate(
        actor_rollout_wg=object(),
        envs=object(),
        responses=["propose-positive", "update-zero", "update-negative", "keep"],
        is_train=True,
    )

    assert rewards.tolist() == [0.5, 0.0, -0.25, 0.0]
    assert calls == [
        ("propose-positive", False),
        ("update-zero", False),
        ("update-negative", False),
        ("keep", False),
        ("propose-positive", True),
    ]
    assert results[0]["status"] == "added"
    assert results[0]["utility_approved"] is True
    assert results[0]["bank_mutated"] is True

    for result in results[1:3]:
        assert result["status"] == "utility_rejected"
        assert result["utility_approved"] is False
        assert result["bank_mutated"] is False

    assert results[3]["status"] == "kept"
    assert "utility_approved" not in results[3]
    assert results[3]["bank_mutated"] is False


def test_disabled_utility_keeps_existing_immediate_persistence_behavior():
    collector, calls = _collector_with_utility_rewards([0.0])
    collector._skill_utility_enabled = lambda is_train: False

    results, _, _ = collector._execute_skill_tool_calls_with_utility_gate(
        actor_rollout_wg=object(),
        envs=object(),
        responses=["propose-without-utility"],
        is_train=False,
    )

    assert calls == [("propose-without-utility", True)]
    assert results[0]["status"] == "added"
    assert results[0]["bank_mutated"] is True


def test_positive_utility_gate_controls_real_bank_writes(tmp_path):
    bank_path = tmp_path / "skills.json"
    bank_path.write_text(
        json.dumps({"general_skills": [], "task_specific_skills": {}, "common_mistakes": []}),
        encoding="utf-8",
    )
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector._skill_bank_store = None
    collector._get_skill_bank_path = lambda: str(bank_path)
    collector._skill_tool_rollout_cfg = lambda: {"autosave": True}
    collector._skill_bank_mutation_enabled = lambda is_train: True
    collector._skill_utility_enabled = lambda is_train: True
    collector._skill_downstream_utility_rewards = lambda *args, **kwargs: (
        np.asarray([0.5, 0.0], dtype=np.float32),
        [{}, {}],
    )
    responses = [
        '<tool_call>{"name":"propose_skill","arguments":{"category":"general","title":"Approved","principle":"Use approved guidance","when_to_apply":"When useful","evidence":"Positive probes"}}</tool_call>',
        '<tool_call>{"name":"propose_skill","arguments":{"category":"general","title":"Rejected","principle":"Use rejected guidance","when_to_apply":"When harmful","evidence":"No probe gain"}}</tool_call>',
    ]

    results, _, _ = collector._execute_skill_tool_calls_with_utility_gate(
        actor_rollout_wg=object(),
        envs=SimpleNamespace(retrieval_memory=None),
        responses=responses,
        is_train=True,
    )

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    assert [skill["title"] for skill in bank["general_skills"]] == ["Approved"]
    assert results[0]["bank_mutated"] is True
    assert results[1]["status"] == "utility_rejected"


def test_positive_utility_persists_update_in_place(tmp_path):
    bank_path = tmp_path / "skills.json"
    bank_path.write_text(
        json.dumps(
            {
                "general_skills": [
                    {
                        "skill_id": "gen_001",
                        "title": "Old title",
                        "principle": "Old guidance",
                        "when_to_apply": "Old condition",
                    }
                ],
                "task_specific_skills": {},
                "common_mistakes": [],
            }
        ),
        encoding="utf-8",
    )
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector._skill_bank_store = None
    collector._get_skill_bank_path = lambda: str(bank_path)
    collector._skill_tool_rollout_cfg = lambda: {"autosave": True}
    collector._skill_bank_mutation_enabled = lambda is_train: True
    collector._skill_utility_enabled = lambda is_train: True
    collector._skill_downstream_utility_rewards = lambda *args, **kwargs: (
        np.asarray([0.25], dtype=np.float32),
        [{}],
    )
    response = '<tool_call>{"name":"update_skill","arguments":{"skill_id":"gen_001","title":"New title","principle":"Improved guidance","when_to_apply":"New condition","reason":"Positive probes"}}</tool_call>'

    results, _, _ = collector._execute_skill_tool_calls_with_utility_gate(
        actor_rollout_wg=object(),
        envs=SimpleNamespace(retrieval_memory=None),
        responses=[response],
        is_train=True,
    )

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    assert len(bank["general_skills"]) == 1
    assert bank["general_skills"][0]["title"] == "New title"
    assert results[0]["status"] == "updated"
    assert results[0]["utility_approved"] is True
