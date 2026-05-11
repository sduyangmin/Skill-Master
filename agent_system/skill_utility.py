"""Probe selection utilities for skill-management downstream utility rewards."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_system.alfworld_task_utils import infer_alfworld_task_family_from_gamefile


ProbeType = Literal["same", "different"]


@dataclass(frozen=True)
class ProbeTask:
    env_name: str
    task_id: str
    task_payload: dict[str, Any]
    probe_type: ProbeType
    family: str | None = None


@dataclass(frozen=True)
class ProbeBatch:
    same: list[ProbeTask]
    different: list[ProbeTask]

    def to_jsonable(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "same": [task.__dict__ for task in self.same],
            "different": [task.__dict__ for task in self.different],
        }


class ALFWorldProbeSelector:
    """Select same-family and different-family ALFWorld probe gamefiles."""

    def __init__(
        self,
        gamefiles_path: str,
        *,
        same_k: int = 2,
        different_k: int = 2,
        seed: int = 0,
    ) -> None:
        self.gamefiles_path = str(gamefiles_path)
        self.same_k = int(same_k)
        self.different_k = int(different_k)
        self.seed = int(seed)
        self.records = self._load_records(Path(gamefiles_path))
        self.records_by_family = self._group_by_family(self.records)

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"ALFWorld probe gamefiles file must contain a list: {path}")

        normalized: list[dict[str, Any]] = []
        for idx, record in enumerate(records):
            if isinstance(record, str):
                gamefile = record
                source_index = idx
            elif isinstance(record, dict):
                gamefile = str(record.get("gamefile") or "")
                source_index = record.get("index", idx)
            else:
                continue
            if not gamefile:
                continue
            family = infer_alfworld_task_family_from_gamefile(gamefile)
            normalized.append(
                {
                    "index": source_index,
                    "gamefile": gamefile,
                    "family": family,
                    "split": str(record.get("split") or "").strip() if isinstance(record, dict) else "",
                }
            )
        if not normalized:
            raise ValueError(f"No usable ALFWorld probe gamefiles found in {path}")
        return normalized

    @staticmethod
    def _group_by_family(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            family = record.get("family")
            if not family:
                continue
            grouped.setdefault(str(family), []).append(record)
        return grouped

    def _rng_for(self, gamefile: str, family: str | None) -> random.Random:
        payload = f"{self.seed}|{family or ''}|{gamefile}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return random.Random(int(digest[:16], 16))

    @staticmethod
    def _to_probe(record: dict[str, Any], probe_type: ProbeType) -> ProbeTask:
        gamefile = str(record["gamefile"])
        family = record.get("family")
        return ProbeTask(
            env_name="alfworld",
            task_id=gamefile,
            task_payload={
                "gamefile": gamefile,
                "index": record.get("index"),
                "split": record.get("split"),
            },
            probe_type=probe_type,
            family=str(family) if family else None,
        )

    def select(self, *, gamefile: str, family: str | None = None) -> ProbeBatch:
        family = family or infer_alfworld_task_family_from_gamefile(gamefile)
        rng = self._rng_for(gamefile, family)

        same_candidates = [
            record for record in self.records_by_family.get(str(family), [])
            if record.get("gamefile") != gamefile
        ] if family else []
        different_candidates = [
            record for record in self.records
            if record.get("family") != family and record.get("gamefile") != gamefile
        ]

        same = rng.sample(same_candidates, k=min(self.same_k, len(same_candidates)))
        different = rng.sample(different_candidates, k=min(self.different_k, len(different_candidates)))
        return ProbeBatch(
            same=[self._to_probe(record, "same") for record in same],
            different=[self._to_probe(record, "different") for record in different],
        )
