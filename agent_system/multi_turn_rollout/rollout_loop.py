# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

# NOTE FOR CURRENT TRAINING / DATA CONSTRUCTION:
# For all three tasks, the current rollout and dataset pipeline should first be
# aligned to the turn-level skill-management prompt, not the trajectory-level
# full-history context. In the turn-level path, the final skill-management
# prompt is built from retrieved skills + final environment feedback +
# observation/action episode trace, and the trace intentionally excludes prior
# <think> text. Do not assume the trajectory-level context behavior applies
# when building or training the current turn-level setup.

import json
import os
import re
import tempfile
import copy
import contextlib
from datetime import datetime

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout import skill_management, skill_utility_reward
from agent_system.multi_turn_rollout.skill_dataset_utils import (
    clean_alfworld_observation_text,
    clean_searchqa_observation_text as _clean_searchqa_observation_text,
    clean_webshop_observation_text,
    format_final_webshop_feedback_for_review,
    format_final_environment_feedback_for_review,
)
from typing import Any, List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.tools.skill_bank_tools import SkillBankStore


SKILL_MANAGEMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_skill",
            "description": "Propose and add a new reusable skill to the skill bank after a completed episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Skill category."},
                    "title": {"type": "string", "description": "Short skill title."},
                    "principle": {"type": "string", "description": "Reusable principle."},
                    "when_to_apply": {"type": "string", "description": "Trigger condition."},
                    "evidence": {"type": "string", "description": "Episode evidence."},
                },
                "required": ["category", "title", "principle", "when_to_apply", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_skill",
            "description": "Update an existing skill in the skill bank after a completed episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Existing skill_id or exact retrieved skill title."},
                    "title": {"type": "string", "description": "Updated title."},
                    "principle": {"type": "string", "description": "Updated principle."},
                    "when_to_apply": {"type": "string", "description": "Updated trigger condition."},
                    "reason": {"type": "string", "description": "Why the skill should be revised."},
                },
                "required": ["skill_id", "title", "principle", "when_to_apply", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keep_skill",
            "description": "Keep the skill bank unchanged after a completed episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why no new skill or update is needed."},
                },
                "required": ["reason"],
            },
        },
    },
]

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._skill_bank_store = None
        self._alfworld_probe_selector = None
        self._probe_env_managers = {}
        self._validation_dump_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._validation_dump_run_dirs: dict[tuple[str, str], str] = {}
        self._skill_bank_runtime_id = uuid.uuid4().hex[:8]
        self._active_skill_bank_path: str | None = None
        self._train_mutable_skill_bank_path: str | None = None
        self._validation_mutable_skill_bank_path: str | None = None
        self._validation_mutation_session_id: str | None = None

    def _sanitize_path_component(self, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            return "default"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
        sanitized = sanitized.strip("._-")
        return sanitized or "default"

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = [{
            "content": obs_content,
            "role": "user",
        }]

        obs_tools = obs.get('tools', None)
        tools = obs_tools[item] if obs_tools is not None else None
        
        # Apply chat template
        if tools:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat
        
        return row_dict

    def _skill_tool_rollout_cfg(self):
        return self.config.env.get('skill_tool_rollout', {})

    def _skill_tool_rollout_enabled(self, is_train: bool) -> bool:
        cfg = self._skill_tool_rollout_cfg()
        if not cfg.get('enable', False):
            return False
        return is_train or cfg.get('apply_on_validation', False)

    def _trajectory_level_rollout_enabled(self, is_train: bool) -> bool:
        cfg = self._skill_tool_rollout_cfg()
        if not cfg.get('trajectory_level', False):
            return False
        return is_train or cfg.get('trajectory_level_apply_on_validation', cfg.get('apply_on_validation', False))

    def _skill_bank_mutation_enabled(self, is_train: bool) -> bool:
        cfg = self._skill_tool_rollout_cfg()
        if is_train:
            return bool(cfg.get("mutate_on_train", False))
        return bool(cfg.get("mutate_on_validation", True))

    def _validation_rollout_dump_enabled(self, is_train: bool) -> bool:
        return (not is_train) and bool(self._skill_tool_rollout_cfg().get("dump_validation_rollouts", False))

    def _get_configured_skill_bank_path(self) -> str:
        cfg = self._skill_tool_rollout_cfg()
        skill_bank_path = cfg.get('skill_bank_path', None)
        if skill_bank_path:
            return skill_bank_path
        if self.config.env.get('use_skills_only_memory', False):
            return self.config.env.skills_only_memory.skills_json_path
        raise ValueError("env.skill_tool_rollout.skill_bank_path must be set when skills-only memory is disabled")

    def _get_skill_bank_path(self) -> str:
        return self._active_skill_bank_path or self._get_configured_skill_bank_path()

    def _skill_bank_runtime_dir(self) -> str:
        runtime_root = os.environ.get("SKILLRL_TMPDIR", tempfile.gettempdir())
        runtime_dir = os.path.join(runtime_root, "skill_bank_runtime", self._skill_bank_runtime_id)
        os.makedirs(runtime_dir, exist_ok=True)
        return runtime_dir

    def _copy_skill_bank_to_runtime(self, *, phase: str, session_id: str | None = None) -> str:
        src_path = self._get_configured_skill_bank_path()
        runtime_dir = self._skill_bank_runtime_dir()
        base_name = os.path.basename(src_path) or "skills.json"
        stem, ext = os.path.splitext(base_name)
        ext = ext or ".json"
        session_suffix = f"_{self._sanitize_path_component(session_id)}" if session_id else ""
        dst_path = os.path.join(runtime_dir, f"{stem}_{phase}{session_suffix}{ext}")
        with open(src_path, "r", encoding="utf-8") as src_file:
            bank = json.load(src_file)
        with open(dst_path, "w", encoding="utf-8") as dst_file:
            json.dump(bank, dst_file, ensure_ascii=False, indent=2)
        return dst_path

    @staticmethod
    def _load_skill_bank_from_path(skill_bank_path: str) -> dict[str, Any]:
        with open(skill_bank_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _activate_skill_bank_for_envs(self, envs: EnvironmentManagerBase, skill_bank_path: str) -> None:
        self._active_skill_bank_path = skill_bank_path
        retrieval_memory = getattr(envs, "retrieval_memory", None)
        if retrieval_memory is None or not hasattr(retrieval_memory, "skills"):
            return
        retrieval_memory.skills = self._load_skill_bank_from_path(skill_bank_path)
        if hasattr(retrieval_memory, "_skill_embeddings_cache"):
            retrieval_memory._skill_embeddings_cache = None

    def _validation_mutation_session_key(self, gen_batch: DataProto) -> str:
        meta = getattr(gen_batch, "meta_info", {}) or {}
        global_step = meta.get("global_step", "unknown")
        experiment_name = meta.get("experiment_name", "default")
        return f"{experiment_name}_step_{global_step}"

    def _prepare_active_skill_bank(
        self,
        *,
        envs: EnvironmentManagerBase,
        gen_batch: DataProto,
        is_train: bool,
    ) -> None:
        configured_path = self._get_configured_skill_bank_path()

        if is_train:
            if self._skill_bank_mutation_enabled(is_train=True):
                if self._train_mutable_skill_bank_path is None:
                    self._train_mutable_skill_bank_path = self._copy_skill_bank_to_runtime(phase="train")
                active_path = self._train_mutable_skill_bank_path
            else:
                active_path = configured_path
        else:
            if self._skill_bank_mutation_enabled(is_train=False):
                session_id = self._validation_mutation_session_key(gen_batch)
                if self._validation_mutation_session_id != session_id:
                    self._validation_mutation_session_id = session_id
                    self._validation_mutable_skill_bank_path = self._copy_skill_bank_to_runtime(
                        phase="validation",
                        session_id=session_id,
                    )
                active_path = self._validation_mutable_skill_bank_path
            else:
                self._validation_mutation_session_id = None
                self._validation_mutable_skill_bank_path = None
                active_path = configured_path

        self._activate_skill_bank_for_envs(envs, active_path)

    def _get_skill_bank_store(self) -> SkillBankStore:
        skill_bank_path = self._get_skill_bank_path()
        autosave = self._skill_tool_rollout_cfg().get('autosave', True)
        if self._skill_bank_store is None or self._skill_bank_store.skill_bank_path != skill_bank_path:
            self._skill_bank_store = SkillBankStore(skill_bank_path=skill_bank_path, autosave=autosave)
        return self._skill_bank_store

    def _skill_utility_cfg(self) -> Dict[str, Any]:
        cfg = self._skill_tool_rollout_cfg()
        return cfg.get("utility_reward", {}) or {}

    def _skill_utility_enabled(self, is_train: bool = True) -> bool:
        cfg = self._skill_utility_cfg()
        if not bool(cfg.get("enable", False)):
            return False
        if is_train:
            return True
        return bool(cfg.get("apply_on_validation", True))

    def _get_alfworld_probe_selector(self):
        return skill_utility_reward.get_alfworld_probe_selector(self)

    def _get_alfworld_probe_env_manager(self, batch_size: int) -> EnvironmentManagerBase:
        return skill_utility_reward.get_alfworld_probe_env_manager(self, batch_size)

    def _select_alfworld_skill_utility_probes(
        self,
        envs: EnvironmentManagerBase,
        item: int,
    ) -> dict[str, Any]:
        return skill_utility_reward.select_alfworld_skill_utility_probes(self, envs, item)

    def _skill_downstream_utility_rewards(
        self,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        responses: List[str],
        tool_results: List[dict[str, Any]],
        is_train: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        return skill_utility_reward.compute_skill_downstream_utility_rewards(
            self,
            actor_rollout_wg,
            envs,
            responses,
            tool_results,
            is_train=is_train,
        )

    def _temporary_alfworld_skill_bank(
        self,
        envs: EnvironmentManagerBase,
        skill_bank_path: str,
    ):
        return skill_utility_reward.temporary_alfworld_skill_bank(self, envs, skill_bank_path)

    def _apply_skill_mutation_to_bank_copy(
        self,
        *,
        base_skill_bank_path: str,
        response: str,
        tool_result: dict[str, Any],
        envs: EnvironmentManagerBase,
        item: int,
    ) -> tuple[str | None, dict[str, Any]]:
        return skill_utility_reward.apply_skill_mutation_to_bank_copy(
            self,
            base_skill_bank_path=base_skill_bank_path,
            response=response,
            tool_result=tool_result,
            envs=envs,
            item=item,
        )

    def _build_probe_gen_batch(self, probe_tasks: list[dict[str, Any]]) -> DataProto:
        return skill_utility_reward.build_probe_gen_batch(self, probe_tasks)

    def _run_probe_action_rollout(
        self,
        *,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        max_steps: int,
    ) -> dict[str, float]:
        return skill_utility_reward.run_probe_action_rollout(
            self,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            max_steps=max_steps,
        )

    def _probe_score_from_metrics(self, metrics: dict[str, float]) -> float:
        return skill_utility_reward.probe_score_from_metrics(self, metrics)

    def _evaluate_alfworld_probe_batch(
        self,
        *,
        actor_rollout_wg,
        skill_bank_path: str,
        probe_info: dict[str, Any],
    ) -> dict[str, Any]:
        return skill_utility_reward.evaluate_alfworld_probe_batch(
            self,
            actor_rollout_wg=actor_rollout_wg,
            skill_bank_path=skill_bank_path,
            probe_info=probe_info,
        )

    @staticmethod
    def _infer_alfworld_task_category(task: str, gamefile: str | None = None) -> str:
        return skill_management.infer_alfworld_task_category(task, gamefile=gamefile)

    @staticmethod
    def _infer_alfworld_success_family(task: str, gamefile: str | None = None) -> str:
        return skill_management.infer_alfworld_success_family(task, gamefile=gamefile)

    @staticmethod
    def _get_alfworld_gamefile(envs: EnvironmentManagerBase, item: int) -> str | None:
        return skill_management.get_alfworld_gamefile(envs, item)

    @staticmethod
    def _get_webshop_goal_idx(envs: EnvironmentManagerBase, item: int) -> int | None:
        return skill_management.get_webshop_goal_idx(envs, item)

    @staticmethod
    def _get_webshop_goal(envs: EnvironmentManagerBase, item: int) -> str | None:
        return skill_management.get_webshop_goal(envs, item)

    def _infer_skill_category_for_env(self, envs: EnvironmentManagerBase, item: int, task: str) -> str | None:
        return skill_management.infer_expected_skill_category_for_env(
            config=self.config,
            envs=envs,
            item=item,
            task=task,
        )

    def _env_specific_dump_fields(self, envs: EnvironmentManagerBase, item: int, task: str) -> Dict[str, Any]:
        return skill_management.env_specific_dump_fields(
            config=self.config,
            envs=envs,
            item=item,
            task=task,
        )

    @staticmethod
    def _truncate_text(text: Any, max_chars: int) -> str:
        text = str(text)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [TrajectoryCollector._to_jsonable(item) for item in value.tolist()]
        if isinstance(value, dict):
            return {str(key): TrajectoryCollector._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrajectoryCollector._to_jsonable(item) for item in value]
        return value

    @staticmethod
    def _raw_prompt_to_text(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, list) and raw_prompt:
            last = raw_prompt[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
        if isinstance(raw_prompt, dict):
            return str(raw_prompt.get("content", ""))
        return str(raw_prompt or "")

    def _extract_action_from_response(self, response: Any) -> str:
        response = str(response or "")
        # SearchQA / search env: extract <search> or <answer>
        if skill_management.is_search_env(getattr(self, "config", None)):
            search = re.search(r"<search>(.*?)</search>", response, flags=re.DOTALL)
            if search:
                return f"<search>{search.group(1).strip()}</search>"
            answer = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
            if answer:
                return f"<answer>{answer.group(1).strip()}</answer>"
            return response.strip()
        match = re.search(r"<action>\s*(.*?)\s*</action>", response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    @staticmethod
    def _response_declares_completion(response: Any) -> bool:
        response = str(response or "")
        return bool(
            re.search(
                r"Task complete|Task completed|all goals satisfied|Goal satisfied",
                response,
                flags=re.IGNORECASE,
            )
        )

    def _detect_rollout_guard_issue(self, response: str) -> tuple[bool, str]:
        guard_hit, reason, _ = self._detect_rollout_guard_issue_with_debug(response)
        return guard_hit, reason

    def _detect_rollout_guard_issue_with_debug(self, response: str) -> tuple[bool, str, dict[str, Any]]:
        response = str(response or "").strip()
        lowered = response.lower()
        strict_action_tag_guard = bool(self._skill_tool_rollout_cfg().get("strict_action_tag_guard", False))
        required_tags = ("<think>", "</think>", "<action>", "</action>")
        debug = {
            "strict_action_tag_guard": strict_action_tag_guard,
            "response_length": len(response),
            "response_prefix": response[:200],
            "required_tag_presence": {tag: (tag in lowered) for tag in required_tags},
        }
        if not response:
            return True, "empty_response", debug

        if "<th><th>" in lowered or (lowered.startswith("<th") and not lowered.startswith("<think>")):
            return True, "malformed_think_prefix", debug

        action = self._extract_action_from_response(response).strip().lower()
        debug["extracted_action"] = action
        if strict_action_tag_guard:
            if any(tag not in lowered for tag in required_tags):
                return True, "missing_required_tags", debug

        if self._response_declares_completion(response) and action != "done":
            return True, "false_completion_without_done", debug

        return False, "", debug

    def _action_format_penalty_info(self, response: str, response_token_count: int | None = None) -> dict[str, Any]:
        cfg = self._skill_tool_rollout_cfg()
        response = str(response or "").strip()
        lowered = response.lower()

        max_response_length = int(getattr(self.config.data, "max_response_length", 0) or 0)
        clipped = bool(response_token_count is not None and max_response_length > 0 and int(response_token_count) >= max_response_length)
        missing_think_close = "<think>" in lowered and "</think>" not in lowered
        missing_action = "<action>" not in lowered or "</action>" not in lowered
        empty_response = not response

        repetition_hit = False
        repetition_patterns = (
            "i'm stuck",
            "i am stuck",
            "maybe",
            "but ",
            "i think",
            "perhaps",
            "nothing happens",
        )
        for pattern in repetition_patterns:
            if lowered.count(pattern) >= int(cfg.get("action_repetition_threshold", 8)):
                repetition_hit = True
                break

        penalty = 0.0
        if clipped:
            penalty += float(cfg.get("action_response_clip_penalty", 0.0))
        if missing_action:
            penalty += float(cfg.get("action_missing_action_penalty", 0.0))
        if missing_think_close:
            penalty += float(cfg.get("action_missing_think_close_penalty", 0.0))
        if empty_response:
            penalty += float(cfg.get("action_empty_response_penalty", 0.0))
        if repetition_hit:
            penalty += float(cfg.get("action_repetition_penalty", 0.0))

        return {
            "action_format_penalty": float(penalty),
            "action_response_clipped": clipped,
            "action_missing_action": missing_action,
            "action_missing_think_close": missing_think_close,
            "action_empty_response": empty_response,
            "action_repetition": repetition_hit,
            "action_response_token_count": int(response_token_count or 0),
        }

    def _apply_rollout_guard(
        self,
        *,
        responses: list[str],
        response_token_counts: np.ndarray | None = None,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
        cfg = self._skill_tool_rollout_cfg()
        if not cfg.get("malformed_rollout_guard", True):
            empty_flags = np.zeros(len(responses), dtype=bool)
            empty_reasons = np.array([""] * len(responses), dtype=object)
            return rewards, dones, infos, empty_flags, empty_reasons

        penalty = float(cfg.get("malformed_rollout_penalty", 1.0))
        triggered = np.zeros(len(responses), dtype=bool)
        reasons = np.array([""] * len(responses), dtype=object)
        updated_infos: list[dict[str, Any]] = []

        for i, response in enumerate(responses):
            info = dict(infos[i]) if infos[i] is not None else {}
            token_count = None
            if response_token_counts is not None:
                token_count = int(response_token_counts[i])
            info.update(self._action_format_penalty_info(response, token_count))
            guard_hit, reason, guard_debug = self._detect_rollout_guard_issue_with_debug(response)
            if guard_hit:
                rewards[i] = rewards[i] - penalty
                dones[i] = True
                triggered[i] = True
                reasons[i] = reason
                info["is_action_valid"] = False
                info["rollout_guard_triggered"] = True
                info["rollout_guard_reason"] = reason
            else:
                info["rollout_guard_triggered"] = False
                info["rollout_guard_reason"] = ""
            info["rollout_guard_debug"] = guard_debug
            updated_infos.append(info)

        return rewards, dones, updated_infos, triggered, reasons

    def _step_observation_text(self, step_data: Dict[str, Any]) -> str:
        obs_text = step_data.get("anchor_obs", None)
        if not obs_text:
            obs_text = step_data.get("prompt_text", None)
            if obs_text is not None:
                obs_text = self._extract_observation_for_trajectory(obs_text)
        else:
            obs_text = self._extract_observation_for_trajectory(obs_text)
        if obs_text is None:
            obs_text = self._raw_prompt_to_text(step_data.get("raw_prompt", ""))
            obs_text = self._extract_observation_for_trajectory(obs_text)
        return str(obs_text or "")

    def _serialize_rollout_step(self, step_idx: int, step_data: Dict[str, Any]) -> Dict[str, Any]:
        decoded_response = str(step_data.get("decoded_response", "") or "")
        is_skill_turn = bool(step_data.get("is_skill_management_turn", False))
        serialized = {
            "step_idx": step_idx,
            "step_type": "skill_management" if is_skill_turn else "env_action",
            "observation": self._step_observation_text(step_data),
            "action": self._extract_action_from_response(decoded_response),
            "decoded_response": decoded_response,
            "reward": self._to_jsonable(step_data.get("rewards", 0.0)),
            "is_action_valid": self._to_jsonable(step_data.get("is_action_valid", True)),
            "active_masks": self._to_jsonable(step_data.get("active_masks", True)),
        }
        if "prompt_text" in step_data:
            serialized["prompt_text"] = str(step_data.get("prompt_text", "") or "")
        if "prompt_debug" in step_data:
            serialized["prompt_debug"] = self._to_jsonable(step_data.get("prompt_debug"))
        if "rollout_guard_triggered" in step_data:
            serialized["rollout_guard_triggered"] = self._to_jsonable(step_data.get("rollout_guard_triggered"))
        if "rollout_guard_reason" in step_data:
            serialized["rollout_guard_reason"] = str(step_data.get("rollout_guard_reason", "") or "")
        if "rollout_guard_debug" in step_data:
            serialized["rollout_guard_debug"] = self._to_jsonable(step_data.get("rollout_guard_debug"))
        if is_skill_turn:
            serialized["skill_tool_name"] = str(step_data.get("skill_tool_name", "") or "")
            serialized["skill_utility_reward"] = self._to_jsonable(step_data.get("skill_utility_reward", 0.0))
            skill_tool_result = step_data.get("skill_tool_result", "")
            serialized["skill_tool_result"] = str(skill_tool_result or "")
            try:
                serialized["skill_tool_result_parsed"] = json.loads(skill_tool_result)
            except Exception:
                pass
            skill_utility_probe_info = step_data.get("skill_utility_probe_info", "")
            serialized["skill_utility_probe_info"] = str(skill_utility_probe_info or "")
            try:
                serialized["skill_utility_probe_info_parsed"] = json.loads(skill_utility_probe_info)
            except Exception:
                pass
        return serialized

    def _validation_rollout_dump_dir(self, meta_info: Dict[str, Any]) -> str:
        cfg = self._skill_tool_rollout_cfg()
        dump_root = cfg.get("validation_dump_dir", None)
        if dump_root is None:
            dump_root = os.path.join(os.getcwd(), "debug", "validation_rollouts")

        experiment_name = self._sanitize_path_component(meta_info.get("experiment_name", "default"))
        cache_key = (str(dump_root), experiment_name)
        if cache_key not in self._validation_dump_run_dirs:
            run_dir = os.path.join(
                str(dump_root),
                experiment_name,
                self._validation_dump_run_timestamp,
            )
            self._validation_dump_run_dirs[cache_key] = run_dir
        return self._validation_dump_run_dirs[cache_key]

    def _success_dump_fields(
        self,
        *,
        task: str,
        gamefile: str | None,
        episode_idx: int,
        total_episodes: int,
        success: Dict[str, np.ndarray],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        per_episode: Dict[str, Any] = {}
        aggregates: Dict[str, Any] = {}
        task_family = self._infer_alfworld_success_family(task, gamefile=gamefile)
        task_specific_key = f"{task_family}_success_rate"

        for key, value in success.items():
            value_np = np.asarray(value)
            if value_np.ndim == 0:
                aggregates[key] = self._to_jsonable(value_np.item())
                continue
            if len(value_np) == total_episodes:
                per_episode[key] = self._to_jsonable(value_np[episode_idx])
                continue
            if key == task_specific_key and len(value_np) > 0:
                aggregates[key] = self._to_jsonable(np.mean(value_np))
                continue
            aggregates[key] = self._to_jsonable(np.mean(value_np)) if len(value_np) > 0 else None

        return per_episode, aggregates

    def _maybe_dump_validation_rollouts(
        self,
        *,
        gen_batch: DataProto,
        envs: EnvironmentManagerBase,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
        is_train: bool,
    ) -> None:
        if not self._validation_rollout_dump_enabled(is_train=is_train):
            return

        meta_info = dict(getattr(gen_batch, "meta_info", {}) or {})
        dump_dir = self._validation_rollout_dump_dir(meta_info)
        os.makedirs(dump_dir, exist_ok=True)

        global_step = meta_info.get("global_step", "unknown")
        val_batch_idx = meta_info.get("val_batch_idx", "unknown")
        filename = f"step_{global_step}_batch_{val_batch_idx}.jsonl"
        path = os.path.join(dump_dir, filename)

        tasks = list(getattr(envs, "tasks", [""] * len(total_batch_list)))
        with open(path, "a", encoding="utf-8") as f:
            for i, steps in enumerate(total_batch_list):
                task = tasks[i] if i < len(tasks) else ""
                env_dump_fields = self._env_specific_dump_fields(envs, i, task)
                gamefile = env_dump_fields.get("gamefile")
                per_episode_success, aggregate_success = self._success_dump_fields(
                    task=task,
                    gamefile=gamefile,
                    episode_idx=i,
                    total_episodes=len(total_batch_list),
                    success=success,
                )
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "validation_run_timestamp": self._validation_dump_run_timestamp,
                    "experiment_name": meta_info.get("experiment_name", ""),
                    "global_step": global_step,
                    "val_batch_idx": val_batch_idx,
                    "traj_uid": str(traj_uid[i]),
                    "task": str(task),
                    "episode_reward": self._to_jsonable(episode_rewards[i]),
                    "episode_length": self._to_jsonable(episode_lengths[i]),
                    "tool_callings": self._to_jsonable(tool_callings[i]),
                    "retrieved_skills": self._format_retrieved_skills(envs, i),
                    "success": per_episode_success,
                    "success_aggregates": aggregate_success,
                    "steps": [self._serialize_rollout_step(step_idx=j + 1, step_data=step) for j, step in enumerate(steps)],
                }
                record.update(env_dump_fields)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[ValidationRolloutDump] wrote {len(total_batch_list)} episodes to {path}")

    def _format_episode_trace(self, trajectory: List[Dict]) -> str:
        cfg = self._skill_tool_rollout_cfg()
        max_steps = cfg.get('trace_max_steps', 50)
        obs_max_chars = cfg.get('trace_obs_max_chars', 900)
        action_max_chars = cfg.get('trace_action_max_chars', cfg.get('trace_response_max_chars', 220))

        selected = trajectory[-max_steps:]
        lines = []
        start_step = max(1, len(trajectory) - len(selected) + 1)
        for offset, step_data in enumerate(selected, start=start_step):
            obs_text = step_data.get('anchor_obs', None)
            if not obs_text:
                obs_text = step_data.get('prompt_text', None)
                if obs_text is not None:
                    obs_text = self._extract_observation_for_trajectory(obs_text)
            else:
                obs_text = self._extract_observation_for_trajectory(obs_text)
            if obs_text is None:
                obs_text = self._raw_prompt_to_text(step_data.get('raw_prompt', ''))
                obs_text = self._extract_observation_for_trajectory(obs_text)
            response = step_data.get('decoded_response', '')
            if not response and 'responses' in step_data:
                response = self.tokenizer.decode(step_data['responses'], skip_special_tokens=True)
            action = self._extract_action_from_response(response)
            lines.append(
                f"Step {offset}\n"
                f"Observation:\n{self._clean_observation_for_skill_review(obs_text, max_chars=obs_max_chars)}\n"
                f"Action Taken: {self._truncate_text(action, action_max_chars)}"
            )
        return "\n\n".join(lines) if lines else "No episode trajectory was recorded."

    def _format_retrieved_skills(self, envs: EnvironmentManagerBase, item: int) -> str:
        retrieval_memory = getattr(envs, 'retrieval_memory', None)
        retrieved_memories = getattr(envs, 'retrieved_memories', None)
        if retrieval_memory is None or retrieved_memories is None:
            return "No retrieved skills were available."
        try:
            return retrieval_memory.format_for_prompt(retrieved_memories[item])
        except Exception as exc:
            return f"Could not format retrieved skills: {exc}"

    def _build_skill_management_prompt(
        self,
        envs: EnvironmentManagerBase,
        item: int,
        trajectory: List[Dict],
        episode_reward: float,
        success: Dict[str, np.ndarray],
        include_trajectory_recap: bool = True,
    ) -> str:
        task = getattr(envs, 'tasks', [""])[item]
        category = self._infer_skill_category_for_env(envs, item, task)
        success_value = bool(success.get('success_rate', np.zeros(len(getattr(envs, 'tasks', []))))[item])
        return skill_management.build_skill_management_prompt(
            config=getattr(self, "config", None),
            task=task,
            category=category,
            episode_reward=episode_reward,
            success_value=success_value,
            retrieved_skills_text=self._format_retrieved_skills(envs, item),
            episode_trace=self._format_episode_trace(trajectory) if include_trajectory_recap else "",
            include_trajectory_recap=include_trajectory_recap,
            goal_idx=self._get_webshop_goal_idx(envs, item),
            goal=self._get_webshop_goal(envs, item),
        )

    def _parse_skill_tool_call(self, response: str) -> tuple[str | None, dict[str, Any], str | None]:
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL)
        if match:
            payload = match.group(1).strip()
        else:
            json_match = re.search(r"(\{.*\})", response, flags=re.DOTALL)
            payload = json_match.group(1).strip() if json_match else response.strip()
        try:
            call = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, {}, f"invalid_json: {exc}"
        if not isinstance(call, dict):
            return None, {}, "tool_call_payload_is_not_object"
        name = call.get("name")
        arguments = call.get("arguments", {})
        if not isinstance(name, str) or not name:
            return None, {}, "missing_tool_name"
        if not isinstance(arguments, dict):
            return None, {}, "tool_arguments_are_not_object"
        return name, arguments, None

    def _execute_skill_tool_call(
        self,
        envs: EnvironmentManagerBase,
        response: str,
        item: int | None = None,
        mutate_bank: bool = True,
    ) -> dict[str, Any]:
        store = self._get_skill_bank_store() if mutate_bank else SkillBankStore(
            skill_bank_path=self._get_skill_bank_path(),
            autosave=False,
        )
        name, arguments, error = self._parse_skill_tool_call(response)
        if error is not None:
            return {"status": "parse_error", "message": error, "tool": None, "dry_run": not mutate_bank, "bank_mutated": False}

        if name == "propose_skill":
            required = ["category", "title", "principle", "when_to_apply", "evidence"]
            missing = [field for field in required if not str(arguments.get(field, "")).strip()]
            if missing:
                return {"status": "invalid_arguments", "message": f"missing fields: {missing}", "tool": name, "dry_run": not mutate_bank, "bank_mutated": False}
            expected_category = None
            if item is not None and hasattr(envs, "tasks"):
                tasks = getattr(envs, "tasks", [])
                if item < len(tasks):
                    expected_category = self._infer_skill_category_for_env(envs, item, tasks[item])
            proposed_category = str(arguments["category"]).strip()
            if expected_category is not None and proposed_category not in {"general", expected_category}:
                return {
                    "status": "invalid_arguments",
                    "message": (
                        f"invalid category '{proposed_category}' for task category "
                        f"'{expected_category}'; use '{expected_category}' or 'general'"
                    ),
                    "tool": name,
                    "dry_run": not mutate_bank,
                    "bank_mutated": False,
                }
            result = store.propose_skill(
                category=proposed_category,
                title=arguments["title"],
                principle=arguments["principle"],
                when_to_apply=arguments["when_to_apply"],
            )
            result["evidence"] = str(arguments["evidence"]).strip()
        elif name == "update_skill":
            required = ["skill_id", "title", "principle", "when_to_apply", "reason"]
            missing = [field for field in required if not str(arguments.get(field, "")).strip()]
            if missing:
                return {"status": "invalid_arguments", "message": f"missing fields: {missing}", "tool": name, "dry_run": not mutate_bank, "bank_mutated": False}
            result = store.update_skill(
                skill_id=arguments["skill_id"],
                title=arguments["title"],
                principle=arguments["principle"],
                when_to_apply=arguments["when_to_apply"],
            )
            result["reason"] = str(arguments["reason"]).strip()
        elif name == "keep_skill":
            reason = str(arguments.get("reason", "")).strip()
            if not reason:
                return {"status": "invalid_arguments", "message": "missing fields: ['reason']", "tool": name, "dry_run": not mutate_bank, "bank_mutated": False}
            result = store.keep_skill(reason=reason)
        else:
            return {"status": "unknown_tool", "message": f"Unsupported skill tool: {name}", "tool": name, "dry_run": not mutate_bank, "bank_mutated": False}

        result["tool"] = name
        result["dry_run"] = not mutate_bank
        result["bank_mutated"] = bool(mutate_bank and result.get("status") in {"added", "updated"})
        if not mutate_bank and result.get("status") in {"added", "updated"}:
            result["message"] = f"{result.get('message', '').rstrip()} (dry run; bank not saved).".strip()

        retrieval_memory = getattr(envs, "retrieval_memory", None)
        if mutate_bank and retrieval_memory is not None and hasattr(retrieval_memory, "skills"):
            retrieval_memory.skills = store.load()
            if hasattr(retrieval_memory, "_skill_embeddings_cache"):
                retrieval_memory._skill_embeddings_cache = None
        return result

    @staticmethod
    def _tool_name_looks_like_env_action(name: str) -> bool:
        name_l = str(name or "").strip().lower()
        if not name_l or name_l in {"keep_skill", "propose_skill", "update_skill"}:
            return False
        action_prefixes = (
            "go to ",
            "go_to_",
            "take ",
            "move ",
            "open ",
            "close ",
            "examine ",
            "clean ",
            "heat ",
            "cool ",
            "look",
            "inventory",
            "put ",
            "use ",
            "toggle ",
            "drop ",
        )
        return name_l == "done" or any(name_l.startswith(prefix) for prefix in action_prefixes)

    @staticmethod
    def _has_placeholder_skill_text(response: str, arguments: dict[str, Any]) -> bool:
        response_l = str(response or "").lower()
        if "<th>" in response_l or "<think><think>" in response_l:
            return True

        placeholder_values = {"...", "…", "<th>", "<think>", "</think>", "<tool_call>", "</tool_call>"}
        for value in arguments.values():
            if isinstance(value, str) and value.strip().lower() in placeholder_values:
                return True
        return False

    def _skill_tool_base_reward_shaping(self, tool_result: dict[str, Any]) -> float:
        cfg = self._skill_tool_rollout_cfg()
        status = str(tool_result.get("status", "")).strip()
        if status == "parse_error":
            return float(cfg.get("parse_error_penalty", -1.0))
        if status == "invalid_arguments":
            return float(cfg.get("invalid_arguments_penalty", -0.5))
        if status == "unknown_tool":
            return float(cfg.get("unknown_tool_penalty", -0.5))
        return float(cfg.get("valid_format_bonus", 0.1))

    def _skill_tool_quality_adjustment(self, response: str, tool_result: dict[str, Any]) -> float:
        cfg = self._skill_tool_rollout_cfg()
        response = str(response or "")
        response_l = response.lower()
        adjustment = 0.0

        if "<think>" not in response_l or "</think>" not in response_l:
            adjustment += float(cfg.get("missing_think_penalty", -0.4))
        if "<tool_call>" not in response_l or "</tool_call>" not in response_l:
            adjustment += float(cfg.get("missing_tool_call_penalty", -0.4))

        parsed_name, arguments, _ = self._parse_skill_tool_call(response)
        tool_name = str(tool_result.get("tool") or parsed_name or "").strip()
        if self._tool_name_looks_like_env_action(tool_name):
            adjustment += float(cfg.get("action_like_tool_penalty", -2.0))

        if self._has_placeholder_skill_text(response, arguments):
            adjustment += float(cfg.get("placeholder_penalty", -1.0))

        if str(tool_result.get("status", "")).strip() == "duplicate":
            adjustment += float(cfg.get("duplicate_penalty", -0.25))

        return adjustment

    def _skill_tool_reward_shaping(self, response: str, tool_result: dict[str, Any]) -> float:
        base_reward = self._skill_tool_base_reward_shaping(tool_result)
        quality_adjustment = self._skill_tool_quality_adjustment(response, tool_result)
        return base_reward + quality_adjustment

    def _skill_tool_reward_shapings(self, responses: List[str], tool_results: List[dict[str, Any]]) -> np.ndarray:
        return np.array(
            [self._skill_tool_reward_shaping(response, result) for response, result in zip(responses, tool_results)],
            dtype=np.float32,
        )

    def _encode_chat_turn(
        self,
        role: str,
        content: str,
        add_generation_prompt: bool = False,
        tools: list | None = None,
        include_system: bool = False,
    ) -> list[int]:
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        # When tool schemas are provided, use the tokenizer's chat template so the
        # model can see the structured tool definitions in-context.
        if not include_system and not tools:
            text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return self.tokenizer.encode(text, add_special_tokens=False)

        messages = [{"role": role, "content": content}]
        kwargs = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": True,
            **apply_chat_template_kwargs,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            if role == "tool":
                text = f"<|im_start|>tool\n{content}<|im_end|>\n"
                if add_generation_prompt:
                    text += "<|im_start|>assistant\n"
            else:
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                if add_generation_prompt:
                    text += "<|im_start|>assistant\n"
            return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value.tolist()
        return value

    def _valid_prompt_ids_from_step(self, step_data: Dict) -> list[int]:
        prompt_ids = step_data["prompts"]
        prompt_length = prompt_ids.shape[-1]
        prompt_mask = step_data["attention_mask"][:prompt_length].bool()
        return prompt_ids[prompt_mask].detach().cpu().tolist()

    def _valid_response_ids_from_step(self, step_data: Dict) -> list[int]:
        prompt_length = step_data["prompts"].shape[-1]
        response_mask = step_data["attention_mask"][prompt_length:].bool()
        response_ids = step_data["responses"][response_mask]
        return response_ids.detach().cpu().tolist()

    def _extract_observation_for_trajectory(self, prompt_text: str) -> str:
        text = str(prompt_text or "")

        # SearchQA: extract <information> blocks (search results from previous step)
        if skill_management.is_search_env(getattr(self, "config", None)):
            info_blocks = re.findall(r"<information>.*?</information>", text, flags=re.DOTALL)
            if info_blocks:
                text = "\n".join(block.strip() for block in info_blocks)
            else:
                text = ""  # First step has no information blocks

        markers = [
            "You are now at step",
            "Your current observation is:",
            "Current observation:",
        ]
        start_idx = -1
        for marker in markers:
            start_idx = text.find(marker)
            if start_idx >= 0:
                break
        if start_idx >= 0:
            text = text[start_idx:]

        end_marker = "\nNow it's your turn to take an action."
        end_idx = text.find(end_marker)
        if end_idx >= 0:
            text = text[:end_idx]

        max_chars = self._skill_tool_rollout_cfg().get("trajectory_obs_max_chars", 6000)
        return self._clean_observation_for_skill_review(text, max_chars=max_chars)

    def _clean_observation_for_skill_review(self, text: Any, max_chars: int) -> str:
        if skill_management.is_webshop_env(getattr(self, "config", None)):
            return clean_webshop_observation_text(str(text or ""), max_chars=max_chars)
        if skill_management.is_search_env(getattr(self, "config", None)):
            return _clean_searchqa_observation_text(str(text or ""), max_chars=max_chars)
        return clean_alfworld_observation_text(str(text or ""), max_chars=max_chars)

    def _format_final_feedback_for_skill_review(self, text: Any) -> str:
        max_chars = self._skill_tool_rollout_cfg().get("trajectory_obs_max_chars", 6000)
        if skill_management.is_webshop_env(getattr(self, "config", None)):
            return format_final_webshop_feedback_for_review(str(text or ""), max_chars=max_chars)
        return format_final_environment_feedback_for_review(str(text or ""), max_chars=max_chars)

    def _truncate_token_ids(self, token_ids: list[int], max_length: int, truncation: str) -> list[int]:
        if len(token_ids) <= max_length:
            return token_ids
        if truncation == "left":
            return token_ids[-max_length:]
        if truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            return token_ids[:left_half] + token_ids[-right_half:]
        return token_ids[:max_length]

    def _truncate_context_keep_prefix(self, token_ids: list[int], prefix_ids: list[int], max_length: int) -> list[int]:
        if len(token_ids) <= max_length:
            return token_ids

        # Keep the initial task/skill context and truncate only the rolling
        # interaction history that follows it.
        if not prefix_ids:
            return self._truncate_token_ids(token_ids, max_length, "left")

        if len(prefix_ids) >= max_length:
            return prefix_ids[:max_length]

        tail_budget = max_length - len(prefix_ids)
        tail_ids = token_ids[-tail_budget:]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
        turn_start = tail_text.find("<|im_start|>")
        if turn_start > 0:
            aligned_tail_ids = self.tokenizer.encode(tail_text[turn_start:], add_special_tokens=False)
            if aligned_tail_ids:
                tail_ids = aligned_tail_ids[-tail_budget:]
        return prefix_ids + tail_ids[-tail_budget:]

    def _compose_context_with_step_chunks(
        self,
        prefix_ids: list[int],
        step_chunks: list[list[int]],
        max_length: int,
        suffix_ids: list[int] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        suffix_ids = suffix_ids or []
        mandatory_ids = prefix_ids + suffix_ids
        debug_info: dict[str, Any] = {
            "prompt_builder": "step_chunk_prefix_v1",
            "prefix_frozen_after_step": 1,
            "prefix_token_count": len(prefix_ids),
            "suffix_token_count": len(suffix_ids),
            "available_history_step_chunks": len(step_chunks),
            "kept_history_step_chunks": 0,
            "dropped_history_step_chunks": len(step_chunks),
            "kept_history_step_start": None,
            "kept_history_step_end": None,
            "truncated_mandatory": False,
        }
        if len(mandatory_ids) >= max_length:
            debug_info["truncated_mandatory"] = True
            if not suffix_ids:
                return prefix_ids[:max_length], debug_info

            suffix_budget = min(len(suffix_ids), max_length)
            prefix_budget = max(0, max_length - suffix_budget)
            return prefix_ids[:prefix_budget] + suffix_ids[:suffix_budget], debug_info

        remaining_budget = max_length - len(mandatory_ids)
        kept_chunks: list[list[int]] = []
        used_budget = 0

        # Keep recent complete environment steps. A chunk is one full step:
        # assistant action plus the following user feedback turn when available.
        for chunk in reversed(step_chunks):
            chunk_len = len(chunk)
            if chunk_len > remaining_budget - used_budget:
                break
            kept_chunks.append(chunk)
            used_budget += chunk_len

        kept_chunk_count = len(kept_chunks)
        debug_info["kept_history_step_chunks"] = kept_chunk_count
        debug_info["dropped_history_step_chunks"] = len(step_chunks) - kept_chunk_count
        if kept_chunk_count > 0:
            first_kept_index = len(step_chunks) - kept_chunk_count
            debug_info["kept_history_step_start"] = first_kept_index + 2
            debug_info["kept_history_step_end"] = len(step_chunks) + 1

        kept_chunks.reverse()
        context_ids = list(prefix_ids)
        for chunk in kept_chunks:
            context_ids.extend(chunk)
        context_ids.extend(suffix_ids)
        return context_ids, debug_info

    def _right_pad_tensor(self, token_ids: list[int], max_length: int, pad_value: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
        token_ids = token_ids[:max_length]
        return torch.tensor(token_ids + [pad_value] * (max_length - len(token_ids)), dtype=dtype)

    def _left_pad_tensor(self, token_ids: list[int], max_length: int, pad_value: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
        token_ids = token_ids[-max_length:]
        return torch.tensor([pad_value] * (max_length - len(token_ids)) + token_ids, dtype=dtype)

    def _preprocess_prompt_ids_batch(
        self,
        gen_batch: DataProto,
        prompt_ids_list: list[list[int]],
        prompt_texts: list[str],
        anchor_obs: list[Any] | None = None,
        prompt_debug_infos: list[Dict[str, Any]] | None = None,
    ) -> DataProto:
        max_prompt_length = int(self.config.data.max_prompt_length)
        pad_token_id = self.tokenizer.pad_token_id
        truncation = self.config.data.truncation
        processed_samples = []
        for item, prompt_ids in enumerate(prompt_ids_list):
            raw_prompt_ids = self._truncate_token_ids(prompt_ids, max_prompt_length, truncation if truncation != "error" else "left")
            input_ids = self._left_pad_tensor(raw_prompt_ids, max_prompt_length, pad_token_id)
            attention_mask = self._left_pad_tensor([1] * len(raw_prompt_ids), max_prompt_length, 0)
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
            processed_samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "raw_prompt_ids": raw_prompt_ids,
                    "anchor_obs": anchor_obs[item] if anchor_obs is not None else prompt_texts[item],
                    "index": item,
                    "data_source": gen_batch.non_tensor_batch["data_source"][item],
                    "prompt_text": prompt_texts[item],
                    "prompt_debug": prompt_debug_infos[item] if prompt_debug_infos is not None else {},
                }
            )
        return DataProto.from_single_dict(data=collate_fn(processed_samples), meta_info=gen_batch.meta_info)

    def _build_trajectory_sample_from_tokens(
        self,
        initial_prompt_ids: list[int],
        response_ids: list[int],
        response_attention_mask: list[int],
        response_loss_mask: list[int],
        metadata: Dict[str, Any],
    ) -> Dict:
        cfg = self._skill_tool_rollout_cfg()
        prompt_length = int(self.config.data.max_prompt_length)
        response_length = int(cfg.get("trajectory_response_length", cfg.get("max_response_length", self.config.data.max_response_length)))
        response_truncation = cfg.get("trajectory_response_truncation", "right")
        pad_token_id = self.tokenizer.pad_token_id

        prompt_ids = self._truncate_context_keep_prefix(initial_prompt_ids, initial_prompt_ids, prompt_length)
        if len(response_ids) > response_length:
            if response_truncation == "left":
                response_ids = response_ids[-response_length:]
                response_attention_mask = response_attention_mask[-response_length:]
                response_loss_mask = response_loss_mask[-response_length:]
            elif response_truncation == "middle":
                left_half = response_length // 2
                right_half = response_length - left_half
                response_ids = response_ids[:left_half] + response_ids[-right_half:]
                response_attention_mask = response_attention_mask[:left_half] + response_attention_mask[-right_half:]
                response_loss_mask = response_loss_mask[:left_half] + response_loss_mask[-right_half:]
            else:
                response_ids = response_ids[:response_length]
                response_attention_mask = response_attention_mask[:response_length]
                response_loss_mask = response_loss_mask[:response_length]

        prompt_tensor = self._left_pad_tensor(prompt_ids, prompt_length, pad_token_id)
        prompt_attention_mask = self._left_pad_tensor([1] * len(prompt_ids), prompt_length, 0)
        response_tensor = self._right_pad_tensor(response_ids, response_length, pad_token_id)
        response_attention_tensor = self._right_pad_tensor(response_attention_mask, response_length, 0)
        response_loss_tensor = self._right_pad_tensor(response_loss_mask, response_length, 0)

        input_ids = torch.cat([prompt_tensor, response_tensor], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_tensor], dim=-1)
        loss_mask = torch.cat([torch.zeros_like(prompt_attention_mask), response_loss_tensor], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        sample = {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "raw_prompt_ids": np.array(prompt_ids, dtype=object),
        }
        sample.update(metadata)
        return sample

    def _debug_print_trajectory_samples(self, samples: list[Dict]) -> None:
        cfg = self._skill_tool_rollout_cfg()
        if not cfg.get("debug_print_trajectory", False):
            return
        num_samples = min(int(cfg.get("debug_print_num_samples", 1)), len(samples))
        max_chars = int(cfg.get("debug_print_max_chars", 12000))
        dump_dir = cfg.get("debug_dump_dir", None)
        print_console = bool(cfg.get("debug_print_console", dump_dir is None))
        if dump_dir:
            os.makedirs(str(dump_dir), exist_ok=True)
        for i in range(num_samples):
            sample = samples[i]
            prompt_ids = sample["prompts"][sample["prompts"] != self.tokenizer.pad_token_id]
            response_mask = sample["attention_mask"][-sample["responses"].shape[-1]:].bool()
            response_ids = sample["responses"][response_mask]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            loss_tokens = int(sample["loss_mask"].sum().item())
            valid_tokens = int(sample["attention_mask"].sum().item())
            header = (
                f"[TrajectoryDebug][sample={i}][traj_uid={sample.get('traj_uid', '')}]\n"
                f"[TrajectoryDebug][episode_reward] {sample.get('episode_rewards', '')}\n"
                f"[TrajectoryDebug][episode_lengths] {sample.get('episode_lengths', '')}\n"
                f"[TrajectoryDebug][valid_tokens] {valid_tokens}\n"
                f"[TrajectoryDebug][loss_tokens] {loss_tokens}"
            )
            if dump_dir:
                safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample.get("traj_uid", "unknown")))[:80]
                path = os.path.join(str(dump_dir), f"trajectory_{os.getpid()}_{i}_{safe_uid}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write("\n[TrajectoryDebug][prompt]\n")
                    f.write(prompt_text)
                    f.write("\n[TrajectoryDebug][response]\n")
                    f.write(response_text)
                    f.write("\n")
                print(f"{header}\n[TrajectoryDebug][dump] {path}")
            if not print_console:
                continue
            print(f"[TrajectoryDebug][sample={i}][traj_uid={sample.get('traj_uid', '')}]")
            print(f"[TrajectoryDebug][episode_reward] {sample.get('episode_rewards', '')}")
            print(f"[TrajectoryDebug][episode_lengths] {sample.get('episode_lengths', '')}")
            print(f"[TrajectoryDebug][valid_tokens] {valid_tokens}")
            print(f"[TrajectoryDebug][loss_tokens] {loss_tokens}")
            print("[TrajectoryDebug][prompt]")
            print(self._truncate_text(prompt_text, max_chars))
            print("[TrajectoryDebug][response]")
            print(self._truncate_text(response_text, max_chars))

    def _debug_verify_skill_loss_mask(self, samples: list[Dict]) -> None:
        """Runtime verifier for trajectory-level loss_mask.

        Segments the response-region loss_mask into runs of 1s vs 0s and
        checks each trained region's content against expectations. Enabled
        only when `env.skill_tool_rollout.verify_skill_loss_mask=True`
        (smoke/debug flag). Emits one line per sample with per-segment
        snippets, plus global counters. Assertions are logged rather than
        raised so the run is not aborted.
        """
        cfg = self._skill_tool_rollout_cfg()
        if not cfg.get("verify_skill_loss_mask", False):
            return
        max_samples = int(cfg.get("verify_skill_loss_mask_max_samples", 2))
        snippet_chars = int(cfg.get("verify_skill_loss_mask_snippet_chars", 160))
        pad_token_id = self.tokenizer.pad_token_id

        for i, sample in enumerate(samples[:max_samples]):
            response_len = sample["responses"].shape[-1]
            response_ids = sample["responses"].tolist()
            response_attn = sample["attention_mask"][-response_len:].tolist()
            loss_mask = sample["loss_mask"][-response_len:].tolist()

            trained_tokens = [rid for rid, m in zip(response_ids, loss_mask) if m == 1]
            trained_text = self.tokenizer.decode(trained_tokens, skip_special_tokens=False)

            segments: list[tuple[int, int, int]] = []
            cur_val = None
            cur_start = 0
            for idx, (attn, m) in enumerate(zip(response_attn, loss_mask)):
                if attn == 0:
                    if cur_val is not None:
                        segments.append((cur_val, cur_start, idx))
                        cur_val = None
                    continue
                if cur_val is None:
                    cur_val = m
                    cur_start = idx
                elif m != cur_val:
                    segments.append((cur_val, cur_start, idx))
                    cur_val = m
                    cur_start = idx
            if cur_val is not None:
                pad_boundary = response_len
                for idx in range(response_len - 1, -1, -1):
                    if response_attn[idx]:
                        pad_boundary = idx + 1
                        break
                segments.append((cur_val, cur_start, pad_boundary))

            tool_call_in_trained = "<tool_call>" in trained_text
            tool_call_in_untrained = any(
                val == 0 and "<tool_call>" in self.tokenizer.decode(
                    response_ids[s:e], skip_special_tokens=False
                )
                for val, s, e in segments
            )
            n_trained = sum(1 for val, _, _ in segments if val == 1)
            n_untrained = sum(1 for val, _, _ in segments if val == 0)
            n_trained_tokens = sum(e - s for val, s, e in segments if val == 1)
            n_untrained_tokens = sum(e - s for val, s, e in segments if val == 0)

            print(
                f"[SkillLossMaskVerify][sample={i}] n_segments={len(segments)} "
                f"trained_segs={n_trained} trained_tokens={n_trained_tokens} "
                f"untrained_segs={n_untrained} untrained_tokens={n_untrained_tokens} "
                f"tool_call_in_trained={tool_call_in_trained} "
                f"tool_call_in_untrained={tool_call_in_untrained}"
            )
            if not tool_call_in_trained:
                print(
                    f"[SkillLossMaskVerify][sample={i}][WARN] no <tool_call> found in trained tokens "
                    "— skill-management response may not be receiving gradient."
                )
            for j, (val, s, e) in enumerate(segments):
                snippet = self.tokenizer.decode(response_ids[s:e], skip_special_tokens=False)
                snippet = snippet.replace("\n", "\\n")
                if len(snippet) > snippet_chars:
                    snippet = snippet[: snippet_chars // 2] + " ... " + snippet[-snippet_chars // 2 :]
                print(
                    f"[SkillLossMaskVerify][sample={i}][seg={j}] mask={val} "
                    f"range=[{s}:{e}] len={e - s} text={snippet!r}"
                )

    def _build_trajectory_level_sample(self, steps: List[Dict], episode_reward: float, episode_length: float, success_rate: Dict[str, float], traj_uid: str, tool_callings: float) -> Dict:
        if not steps:
            raise ValueError("Cannot build a trajectory-level sample from an empty episode.")

        prompt_ids = self._valid_prompt_ids_from_step(steps[0])
        prompt_ids = self._truncate_context_keep_prefix(prompt_ids, prompt_ids, int(self.config.data.max_prompt_length))

        response_ids: list[int] = []
        response_attention_mask: list[int] = []
        response_loss_mask: list[int] = []
        decoded_responses: list[str] = []
        tool_observations: list[str] = []

        for step_index, step_data in enumerate(steps):
            assistant_ids = self._valid_response_ids_from_step(step_data)
            response_ids.extend(assistant_ids)
            response_attention_mask.extend([1] * len(assistant_ids))
            response_loss_mask.extend([1] * len(assistant_ids))
            decoded_responses.append(str(step_data.get("decoded_response", "")))

            if "skill_tool_result" in step_data:
                tool_result = str(step_data.get("skill_tool_result", ""))
                if tool_result:
                    tool_observations.append(tool_result)
                    tool_ids = self._encode_chat_turn("tool", tool_result, add_generation_prompt=False)
                    response_ids.extend(tool_ids)
                    response_attention_mask.extend([1] * len(tool_ids))
                    response_loss_mask.extend([0] * len(tool_ids))

            if step_index + 1 < len(steps):
                next_step = steps[step_index + 1]
                next_prompt = self._extract_observation_for_trajectory(next_step.get("prompt_text", ""))
                next_tools = SKILL_MANAGEMENT_TOOLS if "skill_tool_result" in next_step else None
                user_ids = self._encode_chat_turn("user", next_prompt, add_generation_prompt=True, tools=next_tools)
                response_ids.extend(user_ids)
                response_attention_mask.extend([1] * len(user_ids))
                response_loss_mask.extend([0] * len(user_ids))

        first_step = steps[0]
        valid_actions = []
        for step_data in steps:
            if "is_action_valid" in step_data:
                valid_actions.append(bool(self._to_python_scalar(step_data["is_action_valid"])))
        invalid_action_count = float(sum(1 for valid in valid_actions if not valid))
        action_count = float(len(valid_actions))
        valid_action_ratio = float(np.mean(valid_actions)) if valid_actions else 1.0
        action_format_penalty_sum = float(sum(float(self._to_python_scalar(step_data.get("action_format_penalty", 0.0))) for step_data in steps))
        action_response_clip_count = float(sum(1 for step_data in steps if bool(self._to_python_scalar(step_data.get("action_response_clipped", False)))))
        action_missing_action_count = float(sum(1 for step_data in steps if bool(self._to_python_scalar(step_data.get("action_missing_action", False)))))
        action_missing_think_close_count = float(sum(1 for step_data in steps if bool(self._to_python_scalar(step_data.get("action_missing_think_close", False)))))
        action_repetition_count = float(sum(1 for step_data in steps if bool(self._to_python_scalar(step_data.get("action_repetition", False)))))

        sample = self._build_trajectory_sample_from_tokens(
            initial_prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_attention_mask=response_attention_mask,
            response_loss_mask=response_loss_mask,
            metadata={
                "anchor_obs": first_step.get("anchor_obs", ""),
                "index": first_step.get("index", 0),
                "data_source": first_step.get("data_source", "text"),
                "prompt_text": first_step.get("prompt_text", ""),
                "decoded_response": "\n".join(decoded_responses),
                "tool_observations": "\n".join(tool_observations),
                "uid": first_step.get("uid", traj_uid),
                "traj_uid": traj_uid,
                "is_action_valid": np.bool_(invalid_action_count == 0.0),
                "trajectory_action_valid": np.bool_(all(valid_actions) if valid_actions else True),
                "invalid_action_count": invalid_action_count,
                "trajectory_action_count": action_count,
                "trajectory_valid_action_ratio": valid_action_ratio,
                "action_format_penalty_sum": action_format_penalty_sum,
                "action_response_clip_count": action_response_clip_count,
                "action_missing_action_count": action_missing_action_count,
                "action_missing_think_close_count": action_missing_think_close_count,
                "action_repetition_count": action_repetition_count,
                "rewards": episode_reward,
                "active_masks": True,
                "episode_rewards": episode_reward,
                "episode_lengths": episode_length,
                "tool_callings": tool_callings,
            },
        )
        sample.update(success_rate)
        return sample


    def _append_skill_management_turn(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        uid_batch: np.ndarray,
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
        is_train: bool,
        final_feedback_texts: List[str] | None = None,
    ):
        batch_size = len(total_batch_list)
        prompts = []
        for i in range(batch_size):
            final_feedback = (
                final_feedback_texts[i]
                if final_feedback_texts is not None
                else "No final environment feedback was recorded."
            )
            skill_prompt = self._build_skill_management_prompt(
                envs=envs,
                item=i,
                trajectory=total_batch_list[i],
                episode_reward=float(episode_rewards[i]),
                success=success,
            )
            prompts.append(
                "Final environment feedback:\n"
                f"{final_feedback}\n\n"
                f"{skill_prompt}"
            )
        obs = {
            "text": prompts,
            "image": None,
            "anchor": np.array(prompts, dtype=object),
            "tools": [SKILL_MANAGEMENT_TOOLS for _ in range(batch_size)],
        }

        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
        batch.non_tensor_batch['prompt_text'] = np.array(prompts, dtype=object)
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info

        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

        batch.non_tensor_batch['uid'] = uid_batch
        batch.non_tensor_batch['traj_uid'] = traj_uid
        batch = batch.union(batch_output)

        responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
        mutate_bank = self._skill_bank_mutation_enabled(is_train=is_train)
        tool_results = [
            self._execute_skill_tool_call(envs, response, item=i, mutate_bank=mutate_bank)
            for i, response in enumerate(responses)
        ]
        utility_rewards, utility_probe_infos = self._skill_downstream_utility_rewards(
            actor_rollout_wg,
            envs,
            responses,
            tool_results,
            is_train=is_train,
        )
        skill_reward_shaping = self._skill_tool_reward_shapings(responses, tool_results) + utility_rewards
        valid_tools = np.array([result.get("status") not in {"parse_error", "invalid_arguments", "unknown_tool"} for result in tool_results], dtype=bool)

        batch.non_tensor_batch['decoded_response'] = np.array(responses, dtype=object)
        batch.non_tensor_batch['skill_tool_result'] = np.array([json.dumps(result, ensure_ascii=False) for result in tool_results], dtype=object)
        batch.non_tensor_batch['skill_tool_name'] = np.array([result.get("tool") or "" for result in tool_results], dtype=object)
        batch.non_tensor_batch['skill_reward_shaping'] = skill_reward_shaping.astype(object)
        batch.non_tensor_batch['skill_utility_reward'] = utility_rewards.astype(object)
        batch.non_tensor_batch['skill_utility_probe_info'] = np.array(
            [json.dumps(info, ensure_ascii=False) for info in utility_probe_infos],
            dtype=object,
        )
        batch.non_tensor_batch['is_skill_management_turn'] = np.ones(batch_size, dtype=bool)
        batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
        batch.non_tensor_batch['is_skill_tool_valid'] = valid_tools
        batch.non_tensor_batch['rewards'] = skill_reward_shaping.astype(object)
        batch.non_tensor_batch['active_masks'] = np.ones(batch_size, dtype=object)
        # Skill-management turns do not run the action-path rollout guard;
        # pad the guard fields with neutral values so downstream collation
        # sees the same keys/shape as action turns.
        batch.non_tensor_batch['rollout_guard_triggered'] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch['rollout_guard_reason'] = np.array([""] * batch_size, dtype=object)
        batch.non_tensor_batch['rollout_guard_debug'] = np.array([{} for _ in range(batch_size)], dtype=object)
        batch.non_tensor_batch['action_format_penalty'] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch['action_response_clipped'] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch['action_missing_action'] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch['action_missing_think_close'] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch['action_empty_response'] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch['action_repetition'] = np.zeros(batch_size, dtype=bool)
        batch.non_tensor_batch['action_response_token_count'] = np.zeros(batch_size, dtype=np.int32)

        tool_callings += valid_tools.astype(np.float32)

        batch_list: list[dict] = to_list_of_dict(batch)
        for i in range(batch_size):
            total_batch_list[i].append(batch_list[i])

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def gather_trajectory_level_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        success_rate = {key: np.mean(value) for key, value in success.items()}
        effective_batch = []
        for bs, steps in enumerate(total_batch_list):
            if not steps:
                continue
            if not steps[-1].get("active_masks", True):
                continue
            effective_batch.append(
                self._build_trajectory_level_sample(
                    steps=steps,
                    episode_reward=episode_rewards[bs],
                    episode_length=episode_lengths[bs],
                    success_rate=success_rate,
                    traj_uid=traj_uid[bs],
                    tool_callings=tool_callings[bs],
                )
            )

        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        if obs.get("anchor", None) is not None:
            last_env_feedback = [
                self._format_final_feedback_for_skill_review(value)
                for value in obs.get("anchor", [""] * batch_size)
            ]
        else:
            last_env_feedback = [
                self._format_final_feedback_for_skill_review(value)
                for value in obs.get("text", [""] * batch_size)
            ]
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch.non_tensor_batch['prompt_text'] = np.array(obs.get('text', [""] * batch_size), dtype=object)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            raw_text_actions = list(text_actions)
            batch.non_tensor_batch['decoded_response'] = np.array(raw_text_actions, dtype=object)

            # The ALFWorld projection mutates the passed-in action list in place.
            # Keep a separate immutable copy for rollout guard/debugging.
            env_step_actions = list(raw_text_actions)
            next_obs, rewards, dones, infos = envs.step(env_step_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            prompt_length = batch.batch["prompts"].shape[-1]
            response_attention_masks = batch.batch["attention_mask"][:, prompt_length:]
            rewards, dones, infos, guard_triggered, guard_reasons = self._apply_rollout_guard(
                responses=raw_text_actions,
                response_token_counts=np.array(
                    [
                        int(response_attention_masks[i].sum().item())
                        for i in range(batch_size)
                    ],
                    dtype=np.int32,
                ),
                rewards=rewards,
                dones=dones,
                infos=infos,
            )

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch["rollout_guard_triggered"] = guard_triggered
            batch.non_tensor_batch["rollout_guard_reason"] = guard_reasons
            batch.non_tensor_batch["rollout_guard_debug"] = np.array(
                [info.get("rollout_guard_debug", {}) for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch["action_format_penalty"] = np.array(
                [info.get("action_format_penalty", 0.0) for info in infos],
                dtype=np.float32,
            )
            batch.non_tensor_batch["action_response_clipped"] = np.array(
                [info.get("action_response_clipped", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_missing_action"] = np.array(
                [info.get("action_missing_action", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_missing_think_close"] = np.array(
                [info.get("action_missing_think_close", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_empty_response"] = np.array(
                [info.get("action_empty_response", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_repetition"] = np.array(
                [info.get("action_repetition", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_response_token_count"] = np.array(
                [info.get("action_response_token_count", 0) for info in infos],
                dtype=np.int32,
            )

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['is_skill_management_turn'] = np.zeros(batch_size, dtype=bool)
            batch.non_tensor_batch['skill_reward_shaping'] = np.zeros(batch_size, dtype=object)
            batch.non_tensor_batch['is_skill_tool_valid'] = np.zeros(batch_size, dtype=bool)
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            # Action turns don't invoke a skill-management tool; pad the skill-
            # specific fields so the turn-level collation sees uniform keys.
            batch.non_tensor_batch['skill_tool_result'] = np.array([""] * batch_size, dtype=object)
            batch.non_tensor_batch['skill_tool_name'] = np.array([""] * batch_size, dtype=object)
            batch.non_tensor_batch['skill_utility_reward'] = np.zeros(batch_size, dtype=object)
            batch.non_tensor_batch['skill_utility_probe_info'] = np.array([""] * batch_size, dtype=object)

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                if not active_masks[i]:
                    continue
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)

            next_feedback_source = next_obs.get("anchor", None)
            if next_feedback_source is None:
                next_feedback_source = next_obs.get("text", [""] * batch_size)
            for i in range(batch_size):
                if active_masks[i]:
                    last_env_feedback[i] = self._format_final_feedback_for_skill_review(next_feedback_source[i])
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        if self._skill_tool_rollout_enabled(is_train=is_train):
            self._append_skill_management_turn(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                total_batch_list=total_batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                uid_batch=uid_batch,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                is_train=is_train,
                final_feedback_texts=last_env_feedback,
            )

        self._maybe_dump_validation_rollouts(
            gen_batch=gen_batch,
            envs=envs,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
            is_train=is_train,
        )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def trajectory_level_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        batch_size = len(gen_batch.batch)
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        length_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert batch_size == length_obs, f"gen_batch size {batch_size} does not match obs size {length_obs}"
        if obs.get("image", None) is not None:
            raise NotImplementedError("trajectory-level rollout currently supports text-only ALFWorld observations")

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)

        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        invalid_action_counts = np.zeros(batch_size, dtype=np.float32)
        action_counts = np.zeros(batch_size, dtype=np.float32)
        action_format_penalty_sums = np.zeros(batch_size, dtype=np.float32)
        action_response_clip_counts = np.zeros(batch_size, dtype=np.float32)
        action_missing_action_counts = np.zeros(batch_size, dtype=np.float32)
        action_missing_think_close_counts = np.zeros(batch_size, dtype=np.float32)
        action_repetition_counts = np.zeros(batch_size, dtype=np.float32)
        last_env_feedback = [
            self._extract_observation_for_trajectory(str(text))
            for text in obs.get("text", [""] * batch_size)
        ]

        initial_prompt_ids: list[list[int]] = []
        prefix_ids: list[list[int]] = []
        context_ids: list[list[int]] = []
        step_history_chunks: list[list[list[int]]] = [[] for _ in range(batch_size)]
        trajectory_response_ids: list[list[int]] = [[] for _ in range(batch_size)]
        trajectory_response_attention: list[list[int]] = [[] for _ in range(batch_size)]
        trajectory_loss_mask: list[list[int]] = [[] for _ in range(batch_size)]

        for obs_text in obs["text"]:
            prompt_ids = self._encode_chat_turn("user", str(obs_text), add_generation_prompt=True, include_system=True)
            initial_prompt_ids.append(list(prompt_ids))
            prefix_ids.append(list(prompt_ids))
            context_ids.append(list(prompt_ids))

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            is_last_rollout_step = _step + 1 >= self.config.env.max_steps
            prompt_pairs = [
                self._compose_context_with_step_chunks(
                    prefix_ids=prefix_ids[i],
                    step_chunks=step_history_chunks[i],
                    max_length=int(self.config.data.max_prompt_length),
                )
                for i in range(batch_size)
            ]
            prompt_ids_list = [pair[0] for pair in prompt_pairs]
            prompt_texts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
            prompt_debug_infos = []
            for _prompt_ids, debug_info in prompt_pairs:
                prompt_text = self.tokenizer.decode(_prompt_ids, skip_special_tokens=False)
                prompt_debug_infos.append(
                    {
                        **debug_info,
                        "prompt_contains_task_text": "Your task is to:" in prompt_text,
                        "prompt_contains_retrieved_experience": "Retrieved Relevant Experience" in prompt_text,
                        "prompt_im_start_count": prompt_text.count("<|im_start|>"),
                    }
                )
            batch = self._preprocess_prompt_ids_batch(
                gen_batch=gen_batch,
                prompt_ids_list=prompt_ids_list,
                prompt_texts=prompt_texts,
                anchor_obs=last_env_feedback,
                prompt_debug_infos=prompt_debug_infos,
            )

            batch_input = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            raw_text_actions = list(text_actions)
            batch.non_tensor_batch["decoded_response"] = np.array(raw_text_actions, dtype=object)

            # The ALFWorld projection mutates the passed-in action list in place.
            # Keep a separate immutable copy for rollout guard/debugging.
            env_step_actions = list(raw_text_actions)
            next_obs, rewards, dones, infos = envs.step(env_step_actions)
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            prompt_length = batch.batch["prompts"].shape[-1]
            response_attention_masks = batch.batch["attention_mask"][:, prompt_length:]
            rewards, dones, infos, guard_triggered, guard_reasons = self._apply_rollout_guard(
                responses=raw_text_actions,
                response_token_counts=response_attention_masks.sum(dim=1).detach().cpu().numpy().astype(np.int32),
                rewards=rewards,
                dones=dones,
                infos=infos,
            )

            if "is_action_valid" in infos[0]:
                batch.non_tensor_batch["is_action_valid"] = np.array([info["is_action_valid"] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch["rollout_guard_triggered"] = guard_triggered
            batch.non_tensor_batch["rollout_guard_reason"] = guard_reasons
            batch.non_tensor_batch["rollout_guard_debug"] = np.array(
                [info.get("rollout_guard_debug", {}) for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch["action_format_penalty"] = np.array(
                [info.get("action_format_penalty", 0.0) for info in infos],
                dtype=np.float32,
            )
            batch.non_tensor_batch["action_response_clipped"] = np.array(
                [info.get("action_response_clipped", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_missing_action"] = np.array(
                [info.get("action_missing_action", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_missing_think_close"] = np.array(
                [info.get("action_missing_think_close", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_empty_response"] = np.array(
                [info.get("action_empty_response", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_repetition"] = np.array(
                [info.get("action_repetition", False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch["action_response_token_count"] = np.array(
                [info.get("action_response_token_count", 0) for info in infos],
                dtype=np.int32,
            )
            current_action_valids = batch.non_tensor_batch["is_action_valid"].astype(bool)
            action_counts[active_masks] += 1
            invalid_action_counts[active_masks] += (~current_action_valids).astype(np.float32)[active_masks]
            action_format_penalty_sums[active_masks] += batch.non_tensor_batch["action_format_penalty"].astype(np.float32)[active_masks]
            action_response_clip_counts[active_masks] += batch.non_tensor_batch["action_response_clipped"].astype(np.float32)[active_masks]
            action_missing_action_counts[active_masks] += batch.non_tensor_batch["action_missing_action"].astype(np.float32)[active_masks]
            action_missing_think_close_counts[active_masks] += batch.non_tensor_batch["action_missing_think_close"].astype(np.float32)[active_masks]
            action_repetition_counts[active_masks] += batch.non_tensor_batch["action_repetition"].astype(np.float32)[active_masks]

            if "tool_calling" in infos[0]:
                tool_callings[active_masks] += np.array([info["tool_calling"] for info in infos], dtype=np.float32)[active_masks]

            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1
            batch.non_tensor_batch["is_skill_management_turn"] = np.zeros(batch_size, dtype=bool)
            batch.non_tensor_batch["skill_reward_shaping"] = np.zeros(batch_size, dtype=object)
            batch.non_tensor_batch["is_skill_tool_valid"] = np.zeros(batch_size, dtype=bool)
            batch.non_tensor_batch["skill_tool_result"] = np.array([""] * batch_size, dtype=object)
            batch.non_tensor_batch["skill_tool_name"] = np.array([""] * batch_size, dtype=object)
            batch.non_tensor_batch["skill_utility_reward"] = np.zeros(batch_size, dtype=object)
            batch.non_tensor_batch["skill_utility_probe_info"] = np.array([""] * batch_size, dtype=object)
            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

            batch_list = to_list_of_dict(batch)
            prompt_length = batch.batch["prompts"].shape[-1]
            response_attention_masks = batch.batch["attention_mask"][:, prompt_length:]

            for i in range(batch_size):
                if not active_masks[i]:
                    continue
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

                assistant_ids = batch.batch["responses"][i][response_attention_masks[i].bool()].detach().cpu().tolist()
                step_chunk_ids = list(assistant_ids)
                context_ids[i].extend(assistant_ids)
                trajectory_response_ids[i].extend(assistant_ids)
                trajectory_response_attention[i].extend([1] * len(assistant_ids))
                trajectory_loss_mask[i].extend([1] * len(assistant_ids))

                if next_obs.get("text", None) is not None:
                    last_env_feedback[i] = self._extract_observation_for_trajectory(str(next_obs["text"][i]))
                if not dones[i] and not is_last_rollout_step:
                    user_ids = self._encode_chat_turn("user", last_env_feedback[i], add_generation_prompt=True)
                    context_ids[i].extend(user_ids)
                    trajectory_response_ids[i].extend(user_ids)
                    trajectory_response_attention[i].extend([1] * len(user_ids))
                    trajectory_loss_mask[i].extend([0] * len(user_ids))
                    step_chunk_ids.extend(user_ids)

                if _step == 0:
                    # Freeze the prefix after step 1 so truncation preserves the
                    # initial task context plus the first model/environment turn.
                    prefix_ids[i] = list(context_ids[i])
                elif step_chunk_ids:
                    step_history_chunks[i].append(step_chunk_ids)

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )

        if self._skill_tool_rollout_enabled(is_train=is_train):
            skill_prompts = []
            skill_prompt_ids: list[list[int]] = []
            for i in range(batch_size):
                skill_prompt = self._build_skill_management_prompt(
                    envs=envs,
                    item=i,
                    trajectory=total_batch_list[i],
                    episode_reward=float(episode_rewards[i]),
                    success=success,
                    include_trajectory_recap=False,
                )
                skill_prompts.append(
                    "Final environment feedback:\n"
                    f"{self._format_final_feedback_for_skill_review(last_env_feedback[i])}\n\n"
                    f"{skill_prompt}"
                )
                user_ids = self._encode_chat_turn(
                    "user",
                    skill_prompts[i],
                    add_generation_prompt=True,
                    tools=SKILL_MANAGEMENT_TOOLS,
                )
                skill_prompt_ids.append(list(user_ids))
                context_ids[i].extend(user_ids)
                trajectory_response_ids[i].extend(user_ids)
                trajectory_response_attention[i].extend([1] * len(user_ids))
                trajectory_loss_mask[i].extend([0] * len(user_ids))

            prompt_pairs = [
                self._compose_context_with_step_chunks(
                    prefix_ids=prefix_ids[i],
                    step_chunks=step_history_chunks[i],
                    suffix_ids=skill_prompt_ids[i],
                    max_length=int(self.config.data.max_prompt_length),
                )
                for i in range(batch_size)
            ]
            prompt_ids_list = [pair[0] for pair in prompt_pairs]
            prompt_texts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
            prompt_debug_infos = []
            for _prompt_ids, debug_info in prompt_pairs:
                prompt_text = self.tokenizer.decode(_prompt_ids, skip_special_tokens=False)
                prompt_debug_infos.append(
                    {
                        **debug_info,
                        "prompt_contains_task_text": "Your task is to:" in prompt_text,
                        "prompt_contains_retrieved_experience": "Retrieved Relevant Experience" in prompt_text,
                        "prompt_im_start_count": prompt_text.count("<|im_start|>"),
                        "is_skill_management_prompt": True,
                    }
                )
            batch = self._preprocess_prompt_ids_batch(
                gen_batch=gen_batch,
                prompt_ids_list=prompt_ids_list,
                prompt_texts=prompt_texts,
                anchor_obs=skill_prompts,
                prompt_debug_infos=prompt_debug_infos,
            )
            batch_input = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )
            batch_input.meta_info = gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)
            responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            mutate_bank = self._skill_bank_mutation_enabled(is_train=is_train)
            tool_results = [
                self._execute_skill_tool_call(envs, response, item=i, mutate_bank=mutate_bank)
                for i, response in enumerate(responses)
            ]
            utility_rewards, utility_probe_infos = self._skill_downstream_utility_rewards(
                actor_rollout_wg,
                envs,
                responses,
                tool_results,
            )
            skill_reward_shaping = self._skill_tool_reward_shapings(responses, tool_results) + utility_rewards
            valid_tools = np.array(
                [result.get("status") not in {"parse_error", "invalid_arguments", "unknown_tool"} for result in tool_results],
                dtype=bool,
            )
            batch.non_tensor_batch["decoded_response"] = np.array(responses, dtype=object)
            batch.non_tensor_batch["skill_tool_result"] = np.array([json.dumps(result, ensure_ascii=False) for result in tool_results], dtype=object)
            batch.non_tensor_batch["skill_tool_name"] = np.array([result.get("tool") or "" for result in tool_results], dtype=object)
            batch.non_tensor_batch["skill_reward_shaping"] = skill_reward_shaping.astype(object)
            batch.non_tensor_batch["skill_utility_reward"] = utility_rewards.astype(object)
            batch.non_tensor_batch["skill_utility_probe_info"] = np.array(
                [json.dumps(info, ensure_ascii=False) for info in utility_probe_infos],
                dtype=object,
            )
            batch.non_tensor_batch["is_skill_management_turn"] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch["is_skill_tool_valid"] = valid_tools
            batch.non_tensor_batch["rewards"] = skill_reward_shaping.astype(object)
            batch.non_tensor_batch["active_masks"] = np.ones(batch_size, dtype=object)
            invalid_action_counts += (~valid_tools).astype(np.float32)

            batch_list = to_list_of_dict(batch)
            prompt_length = batch.batch["prompts"].shape[-1]
            response_attention_masks = batch.batch["attention_mask"][:, prompt_length:]
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                assistant_ids = batch.batch["responses"][i][response_attention_masks[i].bool()].detach().cpu().tolist()
                context_ids[i].extend(assistant_ids)
                trajectory_response_ids[i].extend(assistant_ids)
                trajectory_response_attention[i].extend([1] * len(assistant_ids))
                trajectory_loss_mask[i].extend([1] * len(assistant_ids))

                tool_result_text = json.dumps(tool_results[i], ensure_ascii=False)
                tool_ids = self._encode_chat_turn("tool", tool_result_text, add_generation_prompt=False)
                context_ids[i].extend(tool_ids)
                trajectory_response_ids[i].extend(tool_ids)
                trajectory_response_attention[i].extend([1] * len(tool_ids))
                trajectory_loss_mask[i].extend([0] * len(tool_ids))

            tool_callings += valid_tools.astype(np.float32)

        self._maybe_dump_validation_rollouts(
            gen_batch=gen_batch,
            envs=envs,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
            is_train=is_train,
        )

        success_rate = {key: np.mean(value) for key, value in success.items()}
        effective_batch = []
        for i in range(batch_size):
            valid_action_ratio = 1.0
            if action_counts[i] > 0:
                valid_action_ratio = float((action_counts[i] - invalid_action_counts[i]) / action_counts[i])
            metadata = {
                "anchor_obs": last_env_feedback[i],
                "index": i,
                "data_source": gen_batch.non_tensor_batch["data_source"][i],
                "prompt_text": self.tokenizer.decode(initial_prompt_ids[i], skip_special_tokens=False),
                "decoded_response": self.tokenizer.decode(trajectory_response_ids[i], skip_special_tokens=False),
                "uid": uid_batch[i],
                "traj_uid": traj_uid[i],
                "is_action_valid": np.bool_(invalid_action_counts[i] == 0.0),
                "trajectory_action_valid": np.bool_(invalid_action_counts[i] == 0.0),
                "invalid_action_count": invalid_action_counts[i],
                "trajectory_action_count": action_counts[i],
                "trajectory_valid_action_ratio": valid_action_ratio,
                "action_format_penalty_sum": action_format_penalty_sums[i],
                "action_response_clip_count": action_response_clip_counts[i],
                "action_missing_action_count": action_missing_action_counts[i],
                "action_missing_think_close_count": action_missing_think_close_counts[i],
                "action_repetition_count": action_repetition_counts[i],
                "rewards": episode_rewards[i],
                "active_masks": True,
                "episode_rewards": episode_rewards[i],
                "episode_lengths": episode_lengths[i],
                "tool_callings": tool_callings[i],
            }
            metadata.update(success_rate)
            effective_batch.append(
                self._build_trajectory_sample_from_tokens(
                    initial_prompt_ids=initial_prompt_ids[i],
                    response_ids=trajectory_response_ids[i],
                    response_attention_mask=trajectory_response_attention[i],
                    response_loss_mask=trajectory_loss_mask[i],
                    metadata=metadata,
                )
            )

        self._debug_print_trajectory_samples(effective_batch)
        self._debug_verify_skill_loss_mask(effective_batch)
        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=True,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        self._prepare_active_skill_bank(envs=envs, gen_batch=gen_batch, is_train=is_train)

        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        if self._trajectory_level_rollout_enabled(is_train=is_train):
            return self.trajectory_level_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        if self._trajectory_level_rollout_enabled(is_train=is_train):
            gen_batch_output: DataProto = self.gather_trajectory_level_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
            )
        else:
            gen_batch_output: DataProto = self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
            )
        
        return gen_batch_output
