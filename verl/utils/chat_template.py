# Copyright 2025 Bytedance Ltd. and/or its affiliates


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        tokenize=True,
    )
    system_prompt = token1[: -(len(token2) - len(token1))]

    token3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
    )
    generation_prompt = token3[len(token1) :]

    return system_prompt, generation_prompt
