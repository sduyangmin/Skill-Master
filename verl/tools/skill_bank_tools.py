# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import shutil
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema


class SkillBankStore:
    """JSON-backed skill bank store with simple duplicate checks."""

    def __init__(self, skill_bank_path: str, autosave: bool = True):
        if not skill_bank_path:
            raise ValueError("skill_bank_path must be provided")
        self.skill_bank_path = skill_bank_path
        self.autosave = autosave

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.skill_bank_path):
            bank = self._empty_bank()
            if self.autosave:
                self.save(bank)
            return bank

        with open(self.skill_bank_path, "r", encoding="utf-8") as f:
            bank = json.load(f)
        return self._ensure_bank_structure(bank)

    def save(self, bank: dict[str, Any]) -> None:
        dirpath = os.path.dirname(self.skill_bank_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if os.path.exists(self.skill_bank_path):
            shutil.copy2(self.skill_bank_path, self.skill_bank_path + ".bak")
        with open(self.skill_bank_path, "w", encoding="utf-8") as f:
            json.dump(self._ensure_bank_structure(bank), f, indent=2, ensure_ascii=False)

    def get_stats(self, bank: dict[str, Any]) -> dict[str, int]:
        bank = self._ensure_bank_structure(bank)
        general = len(bank["general_skills"])
        task_specific = sum(len(skills) for skills in bank["task_specific_skills"].values())
        common_mistakes = len(bank["common_mistakes"])
        return {
            "general": general,
            "task_specific": task_specific,
            "common_mistakes": common_mistakes,
            "total": general + task_specific + common_mistakes,
        }

    def propose_skill(
        self,
        category: str,
        title: str,
        principle: str,
        when_to_apply: str,
    ) -> dict[str, Any]:
        bank = self.load()
        duplicate = self.find_duplicate(bank, title=title, principle=principle)
        if duplicate is not None:
            return {
                "status": "duplicate",
                "message": "A similar skill already exists.",
                "skill": None,
                "duplicate_of": duplicate["skill_id"],
                "bank_stats": self.get_stats(bank),
            }

        new_skill = {
            "skill_id": self.next_dynamic_skill_id(bank),
            "title": title.strip(),
            "principle": principle.strip(),
            "when_to_apply": when_to_apply.strip(),
        }
        self._append_skill(bank, category=category, skill=new_skill)
        if self.autosave:
            self.save(bank)

        return {
            "status": "added",
            "message": "Skill added to bank.",
            "skill": new_skill,
            "duplicate_of": None,
            "bank_stats": self.get_stats(bank),
        }

    def update_skill(
        self,
        skill_id: str,
        title: str,
        principle: str,
        when_to_apply: str,
    ) -> dict[str, Any]:
        bank = self.load()
        located = self.find_skill(bank, skill_id)
        if located is None:
            return {
                "status": "not_found",
                "message": f"Skill id {skill_id} was not found.",
                "old_skill": None,
                "new_skill": None,
                "bank_stats": self.get_stats(bank),
            }

        _, _, skill = located
        old_skill = copy.deepcopy(skill)
        skill["title"] = title.strip()
        skill["principle"] = principle.strip()
        skill["when_to_apply"] = when_to_apply.strip()
        new_skill = copy.deepcopy(skill)
        if self.autosave:
            self.save(bank)

        return {
            "status": "updated",
            "message": "Skill updated successfully.",
            "old_skill": old_skill,
            "new_skill": new_skill,
            "bank_stats": self.get_stats(bank),
        }

    def keep_skill(self, reason: str) -> dict[str, Any]:
        bank = self.load()
        return {
            "status": "kept",
            "message": "Skill bank unchanged.",
            "reason": reason.strip(),
            "bank_stats": self.get_stats(bank),
        }

    def find_duplicate(self, bank: dict[str, Any], title: str, principle: str) -> Optional[dict[str, Any]]:
        title_norm = self._normalize_text(title)
        principle_norm = self._normalize_text(principle)
        for _, _, skill in self.iter_skills(bank):
            if self._normalize_text(skill.get("title", "")) == title_norm:
                return skill
            if self._normalize_text(skill.get("principle", "")) == principle_norm:
                return skill
        return None

    def find_skill(self, bank: dict[str, Any], skill_id: str) -> Optional[tuple[str, Optional[str], dict[str, Any]]]:
        """Find a skill by stable id, falling back to exact title.

        Retrieved skill prompts currently expose titles but not internal ids.
        Accepting exact titles keeps update_skill usable without requiring the
        model to infer hidden identifiers.
        """
        bank = self._ensure_bank_structure(bank)
        for skill in bank["general_skills"]:
            if skill.get("skill_id") == skill_id:
                return "general", None, skill
        for category, skills in bank["task_specific_skills"].items():
            for skill in skills:
                if skill.get("skill_id") == skill_id:
                    return "task_specific", category, skill
        target_title = self._normalize_text(skill_id)
        if target_title:
            for skill in bank["general_skills"]:
                if self._normalize_text(skill.get("title", "")) == target_title:
                    return "general", None, skill
            for category, skills in bank["task_specific_skills"].items():
                for skill in skills:
                    if self._normalize_text(skill.get("title", "")) == target_title:
                        return "task_specific", category, skill
        return None

    def next_dynamic_skill_id(self, bank: dict[str, Any]) -> str:
        max_idx = 0
        for _, _, skill in self.iter_skills(bank):
            skill_id = skill.get("skill_id", "")
            if skill_id.startswith("dyn_"):
                suffix = skill_id.split("dyn_", 1)[1]
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
        return f"dyn_{max_idx + 1:03d}"

    def iter_skills(self, bank: dict[str, Any]):
        bank = self._ensure_bank_structure(bank)
        for skill in bank["general_skills"]:
            yield "general", None, skill
        for category, skills in bank["task_specific_skills"].items():
            for skill in skills:
                yield "task_specific", category, skill

    def _append_skill(self, bank: dict[str, Any], category: str, skill: dict[str, Any]) -> None:
        bank = self._ensure_bank_structure(bank)
        if category == "general":
            bank["general_skills"].append(skill)
        else:
            bank["task_specific_skills"].setdefault(category, []).append(skill)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    @staticmethod
    def _empty_bank() -> dict[str, Any]:
        return {
            "general_skills": [],
            "task_specific_skills": {},
            "common_mistakes": [],
        }

    def _ensure_bank_structure(self, bank: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(bank, dict):
            bank = {}
        bank.setdefault("general_skills", [])
        bank.setdefault("task_specific_skills", {})
        bank.setdefault("common_mistakes", [])
        return bank


class _SkillBankTool(BaseTool):
    """Shared logic for skill bank tools."""

    REQUIRED_CONFIG_KEYS = ("skill_bank_path",)

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        for key in self.REQUIRED_CONFIG_KEYS:
            if not config.get(key):
                raise ValueError(f"Configuration must include '{key}'")
        self.store = SkillBankStore(
            skill_bank_path=config["skill_bank_path"],
            autosave=config.get("autosave", True),
        )
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"last_response": None}
        return instance_id

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    def _validate_required_fields(self, parameters: dict[str, Any], required_fields: list[str]) -> Optional[str]:
        for field in required_fields:
            value = parameters.get(field)
            if not isinstance(value, str) or not value.strip():
                return field
        return None

    def _dump_response(self, instance_id: str, response: dict[str, Any]) -> Tuple[str, float, dict]:
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["last_response"] = response
        return json.dumps(response, ensure_ascii=False), 0.0, response.get("bank_stats", {})


class ProposeSkillTool(_SkillBankTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        missing = self._validate_required_fields(
            parameters,
            required_fields=["category", "title", "principle", "when_to_apply", "evidence"],
        )
        if missing is not None:
            return self._dump_response(
                instance_id,
                {
                    "status": "invalid_arguments",
                    "message": f"Missing required field: {missing}",
                    "skill": None,
                    "duplicate_of": None,
                    "bank_stats": None,
                },
            )

        response = self.store.propose_skill(
            category=parameters["category"],
            title=parameters["title"],
            principle=parameters["principle"],
            when_to_apply=parameters["when_to_apply"],
        )
        response["evidence"] = parameters["evidence"].strip()
        return self._dump_response(instance_id, response)


class UpdateSkillTool(_SkillBankTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        missing = self._validate_required_fields(
            parameters,
            required_fields=["skill_id", "title", "principle", "when_to_apply", "reason"],
        )
        if missing is not None:
            return self._dump_response(
                instance_id,
                {
                    "status": "invalid_arguments",
                    "message": f"Missing required field: {missing}",
                    "old_skill": None,
                    "new_skill": None,
                    "bank_stats": None,
                },
            )

        response = self.store.update_skill(
            skill_id=parameters["skill_id"],
            title=parameters["title"],
            principle=parameters["principle"],
            when_to_apply=parameters["when_to_apply"],
        )
        response["reason"] = parameters["reason"].strip()
        return self._dump_response(instance_id, response)


class KeepSkillTool(_SkillBankTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        missing = self._validate_required_fields(parameters, required_fields=["reason"])
        if missing is not None:
            return self._dump_response(
                instance_id,
                {
                    "status": "invalid_arguments",
                    "message": f"Missing required field: {missing}",
                    "reason": None,
                    "bank_stats": None,
                },
            )

        response = self.store.keep_skill(reason=parameters["reason"])
        return self._dump_response(instance_id, response)
