# Copyright 2024 Bytedance Ltd. and/or its affiliates

from typing import List, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    import numpy as np
    import pandas as pd

    if isinstance(data_item, dict):
        return {
            k: convert_nested_value_to_list_recursive(v)
            for k, v in data_item.items()
            if v is not None
        }
    if isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    if isinstance(data_item, pd.core.series.Series):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item.tolist()]
    if isinstance(data_item, np.ndarray):
        return convert_nested_value_to_list_recursive(data_item.tolist())
    return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Text-only multi-turn SFT dataset with tool-schema injection support.
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"]
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.max_samples = config.get("max_samples", -1)
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)

        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.apply_chat_template_kwargs = multiturn_config.get("apply_chat_template_kwargs", {})

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                import numpy as np

                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = range(self.max_samples)
            self.dataframe = self.dataframe.iloc[list(indices)]

        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

    def __len__(self):
        return len(self.messages)

    @staticmethod
    def _normalize_tools(tools):
        if tools is None or tools is pd.NA:
            return None
        if isinstance(tools, (list, tuple)):
            return tools if len(tools) > 0 else None
        try:
            is_missing = pd.isna(tools)
            if isinstance(is_missing, bool) and is_missing:
                return None
        except (TypeError, ValueError):
            pass
        return tools if tools else None

    def _process_single_message(self, index: int, message: dict, tools=None):
        inputs = self.tokenizer.apply_chat_template(
            [message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **self.apply_chat_template_kwargs,
        )
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            target_mask = torch.ones_like(attention_mask)
            target_mask[: len(self.generation_prompt)] = 0
            loss_mask = torch.zeros_like(attention_mask)
            # The trainer computes next-token loss with logits[:, :-1] and labels[:, 1:],
            # so mask position i controls whether token i+1 is trained.
            loss_mask[:-1] = target_mask[1:]
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask

    def __getitem__(self, item):
        messages = self.messages[item]
        sample_tools = self._normalize_tools(self.tools[item] if self.tools is not None else None)

        input_ids_parts = []
        loss_mask_parts = []
        attention_mask_parts = []

        for i, message in enumerate(messages):
            message = dict(message)
            message_tools = self._normalize_tools(message.pop("tools", None))
            tools = message_tools if message_tools is not None else (sample_tools if i == 0 else None)
            _input_ids, _loss_mask, _attention_mask = self._process_single_message(
                index=i,
                message=message,
                tools=tools,
            )
            input_ids_parts.append(_input_ids)
            loss_mask_parts.append(_loss_mask)
            attention_mask_parts.append(_attention_mask)

        input_ids = torch.cat(input_ids_parts, dim=0)
        loss_mask = torch.cat(loss_mask_parts, dim=0)
        attention_mask = torch.cat(attention_mask_parts, dim=0)

        sequence_length = input_ids.shape[0]
        if self.pad_mode == "right":
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)
                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")
        else:
            if sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        if self.pad_mode == "right":
            position_ids = F.pad(position_ids, (0, self.max_length - len(position_ids)), value=0)[: self.max_length]
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
