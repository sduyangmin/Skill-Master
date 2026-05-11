import json
import re
from typing import Any

from agent_system.alfworld_task_utils import infer_alfworld_skill_category


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


def infer_alfworld_task_category(task: str, gamefile: str | None = None) -> str:
    return infer_alfworld_skill_category(task, gamefile=gamefile)


def truncate_text(text: Any, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def extract_action_from_response(response: Any) -> str:
    response = str(response or "")
    match = re.search(r"<action>\s*(.*?)\s*</action>", response, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: recover a raw WebShop/ALFWorld-style action mention even if
    # the model omitted XML tags or was truncated before closing them.
    action_patterns = [
        r"\b(search\[[^\]\n]*\])",
        r"\b(click\[[^\]\n]*\])",
        r"\b(go to [^\n<]+)",
        r"\b(open [^\n<]+)",
        r"\b(close [^\n<]+)",
        r"\b(take [^\n<]+)",
        r"\b(put [^\n<]+)",
        r"\b(clean [^\n<]+)",
        r"\b(heat [^\n<]+)",
        r"\b(cool [^\n<]+)",
        r"\b(look\b)",
        r"\b(inventory\b)",
        r"\b(done\b)",
    ]
    for pattern in action_patterns:
        fallback = re.search(pattern, response, flags=re.IGNORECASE)
        if fallback:
            return fallback.group(1).strip()
    return response.strip()


def extract_think_block(response: Any) -> str:
    response = str(response or "")
    match = re.search(r"(<think>\s*.*?\s*</think>)", response, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_observation_for_trajectory(prompt_text: str, max_chars: int = 6000) -> str:
    text = str(prompt_text or "")
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
    return truncate_text(text.strip(), max_chars)


def clean_alfworld_observation_text(prompt_text: str, max_chars: int = 6000) -> str:
    """Return only the observation content needed for trajectory review.

    ALFWorld step prompts often append a long admissible-action list.  That is
    useful for action prediction, but it adds little signal for reviewing a
    completed episode and can dominate the skill-management trace.
    """
    text = extract_observation_for_trajectory(prompt_text, max_chars=max_chars)
    if not text:
        return ""

    admissible_marker = "\nYour admissible actions of the current situation are:"
    admissible_idx = text.find(admissible_marker)
    if admissible_idx >= 0:
        text = text[:admissible_idx]

    text = re.sub(
        r"^You are now at step \d+ and your current observation is:\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(r"^Your current observation is:\s*", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"^Current observation:\s*", "", text, count=1, flags=re.DOTALL)
    return truncate_text(text.strip(), max_chars)


def clean_webshop_observation_text(prompt_text: str, max_chars: int = 6000) -> str:
    """Return a compact WebShop page observation for trajectory review.

    WebShop observations are usually ``[SEP]``-delimited page text.  Keep page
    content intact, but remove rollout prompt scaffolding and format separators
    as readable lines for the final skill-management review.
    """
    text = str(prompt_text or "")
    markers = [
        "and your current observation is:",
        "Your current observation is:",
        "Current observation:",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[idx + len(marker):]
            break

    stop_markers = [
        "\nYour admissible actions of the current situation are:",
        "\nNow it's your turn to take an action.",
    ]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]

    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()

    if "[SEP]" in text:
        parts = []
        for part in text.split("[SEP]"):
            cleaned = part.strip().strip("'\"").strip()
            if cleaned:
                parts.append(cleaned)
        text = "\n".join(parts)

    return truncate_text(text.strip(), max_chars)


def clean_searchqa_observation_text(prompt_text: str, max_chars: int = 6000) -> str:
    """Return compact SearchQA information blocks for trajectory review.

    SearchQA observations are ``<information>...</information>`` blocks
    inside the prompt.  Return them concisely without the surrounding
    instruction / skill-bank scaffolding.
    """
    text = str(prompt_text or "")
    if not text.strip():
        return ""
    # If text already contains only <information> blocks, use it as-is.
    # Otherwise search for blocks inside the full prompt.
    if not text.strip().startswith("<information>"):
        blocks = re.findall(r"<information>.*?</information>", text, flags=re.DOTALL)
        if blocks:
            text = "\n".join(block.strip() for block in blocks)
        # else keep text as-is (first step with no info)
    text = text.strip()
    return truncate_text(text, max_chars)


def _next_labeled_value(lines: list[str], label: str) -> str:
    label = str(label or "").strip().lower()
    lowered = [line.strip().lower() for line in lines]
    for idx, line in enumerate(lowered):
        if line == label and idx + 1 < len(lines):
            value = str(lines[idx + 1]).strip()
            if value:
                return value
    return ""


def format_final_webshop_feedback_for_review(prompt_text: str, max_chars: int = 6000) -> str:
    text = clean_webshop_observation_text(prompt_text, max_chars=max_chars)
    if not text:
        return "No final environment feedback was recorded."

    lines = [line.strip() for line in str(text).splitlines() if str(line).strip()]
    lowered_lines = [line.lower() for line in lines]
    lowered_text = "\n".join(lowered_lines)

    purchased = (
        "purchased" in lowered_lines
        or "thank you for shopping with us!" in lowered_text
    )
    asin = _next_labeled_value(lines, "asin")
    options = _next_labeled_value(lines, "options")
    reward = _next_labeled_value(lines, "your score (min 0.0, max 1.0)")

    summary_lines = [f"Purchased: {'yes' if purchased else 'no'}"]
    if asin and asin.lower() != "none":
        summary_lines.append(f"Purchased ASIN: {asin}")
    if options and options.lower() != "none":
        summary_lines.append(f"Purchased options: {options}")
    if reward and reward.lower() != "none":
        summary_lines.append(f"Reward: {reward}")

    if len(summary_lines) > 1 or purchased:
        return truncate_text("\n".join(summary_lines), max_chars)

    return text


def format_final_environment_feedback_for_review(prompt_text: str, max_chars: int = 6000) -> str:
    text = clean_alfworld_observation_text(prompt_text, max_chars=max_chars)
    if not text:
        return "No final environment feedback was recorded."
    return text


def format_episode_trace(
    trajectory: list[dict[str, Any]],
    max_steps: int = 50,
    obs_max_chars: int = 900,
    action_max_chars: int = 220,
) -> str:
    if not trajectory:
        return "No episode trajectory was recorded."

    selected = trajectory[-max_steps:]
    lines = []
    start_step = max(1, len(trajectory) - len(selected) + 1)
    for offset, step in enumerate(selected, start=start_step):
        observation = clean_alfworld_observation_text(
            step.get("observation") or step.get("prompt_text") or "",
            max_chars=obs_max_chars,
        )
        action = (
            step.get("parsed_action")
            or step.get("action")
            or extract_action_from_response(step.get("teacher_response") or step.get("decoded_response"))
        )
        lines.append(
            f"Step {offset}\n"
            f"Observation:\n{observation}\n"
            f"Action Taken: {truncate_text(action, action_max_chars)}"
        )
    return "\n\n".join(lines)


def build_skill_management_prompt(
    *,
    task: str,
    gamefile: str | None = None,
    retrieved_skills_text: str,
    trajectory: list[dict[str, Any]],
    episode_reward: float,
    success_value: bool,
    include_trajectory_recap: bool = True,
) -> str:
    category = infer_alfworld_task_category(task, gamefile=gamefile)
    outcome = "Success" if success_value else "Failure"

    lines = [
        "You are reviewing a completed ALFWorld episode.",
        "Decide whether the skill bank should be updated based on reusable evidence from this episode.",
        "",
        "Rules:",
        "- Call at most one skill-management tool.",
        "- Choose the tool that best reflects whether the episode reveals a reusable lesson that is missing or incorrect in the current bank.",
        "- Use propose_skill only for a genuinely new reusable lesson.",
        "- Use update_skill only when an existing skill should be revised.",
        "- Use keep_skill only when the current retrieved skills already cover the observed strategy or failure pattern well enough.",
        "- Base your decision on the task, episode evidence, and the current retrieved skills.",
        "- Compare the observed pattern against the retrieved skills explicitly; do not choose keep_skill just because the broad task category already has some skills.",
        "- Success alone is not a reason to keep_skill: if a successful episode demonstrates a concise reusable strategy that is not already covered, propose_skill or update_skill.",
        "- Failure alone is not a reason to change the bank: if the failure does not reveal a concrete reusable lesson beyond the current retrieved skills, use keep_skill.",
        "- For failed episodes, treat repeated invalid loops, repeated ineffective actions after 'Nothing happens.', missed visible targets, incorrect subgoal switching, and losing track of required object counts as strong evidence for propose_skill or update_skill unless an existing retrieved skill already states that rule explicitly.",
        "- For successful episodes, prefer propose_skill or update_skill when the success depends on a reusable tactic, ordering rule, or search heuristic that is not already stated explicitly in the retrieved skills.",
        "- Add or revise a skill only if it is generic, concise, and useful for future ALFWorld episodes.",
        "- Do not propose or update a skill that merely restates an already retrieved skill with minor wording changes.",
        "- Only write skills grounded in the current ALFWorld household environment, task, objects, and receptacles.",
        "- Reject any lesson that introduces out-of-domain entities or scenes not supported by the episode evidence, such as outdoor locations, stores, trees/orchards, fictional names, or unrelated objects.",
        "- Do not store task-instance details such as specific object instances, room names, receptacle IDs, or one-off episode narration unless the lesson is clearly reusable.",
        "- Do not include meta-commentary about skill-bank decisions, prompt quality, guidance quality, success/failure labels, or whether the bank should change.",
        "- Do not output placeholders, ellipses, half-finished text, or copied trajectory fragments.",
        "- For propose_skill or update_skill, make the title short and canonical, and make principle/when_to_apply concise reusable rules rather than episode summaries.",
        "- For update_skill, set skill_id to the exact retrieved skill title if no explicit numeric skill_id is shown.",
        "- Do not output an ALFWorld <action>; the episode is already over.",
        "- Do not choose navigation or environment actions such as look, go to, take, move, clean, heat, cool, or done.",
        "- First reason inside <think> </think>, then output exactly one skill-management tool call in JSON.",
        "- Keep <think> extremely short: 1-3 sentences, no bullet points, no long episode recap, and no copied trajectory details.",
        "- The JSON must be enclosed in <tool_call> </tool_call> tags and must be the final content.",
        "- For task-specific skills, set category to the Skill Category shown below.",
        "- Use category=\"general\" only if the lesson is clearly reusable across multiple task types.",
        "",
        "Output requirements:",
        "- Reason inside <think>...</think>, but keep it brief and decision-focused.",
        "- Your final content must be exactly one skill-management tool call wrapped in <tool_call>...</tool_call>.",
        "- Inside <tool_call>...</tool_call>, output only valid JSON with keys 'name' and 'arguments'.",
        "- If you propose or update a skill, output compact reusable text, not long paragraphs.",
        "- Copy argument field names exactly, including required fields such as evidence for propose_skill.",
        "- Do not use markdown code fences.",
        "",
        "Valid output formats:",
        '<think>...</think>\n<tool_call>{"name":"keep_skill","arguments":{"reason":"..."}}</tool_call>',
        f'<think>...</think>\n<tool_call>{{"name":"propose_skill","arguments":{{"category":"{category}","title":"...","principle":"...","when_to_apply":"...","evidence":"..."}}}}</tool_call>',
        '<think>...</think>\n<tool_call>{"name":"update_skill","arguments":{"skill_id":"...","title":"...","principle":"...","when_to_apply":"...","reason":"..."}}</tool_call>',
        "",
        f"Task: {task}",
        f"Skill Category: {category}",
        f"Episode Outcome: {outcome}",
        f"Episode Reward: {episode_reward}",
        "",
        "Retrieved Skills:",
        str(retrieved_skills_text or "No retrieved skills were available."),
        "",
    ]
    if include_trajectory_recap:
        lines.extend([
            "Episode Trajectory:",
            format_episode_trace(trajectory),
            "END EPISODE TRAJECTORY.",
            "",
        ])
    lines.append(
        "Now output only the final skill-management decision using the required <tool_call> JSON format.",
    )
    return "\n".join(lines)


def parse_tool_call_from_response(response: str) -> tuple[str | None, dict[str, Any], str | None]:
    response = str(response or "")
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


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized = []
    if not tool_calls:
        return normalized

    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None and isinstance(tool_call, dict):
            function = tool_call.get("function", {})
        name = getattr(function, "name", None) if function is not None else None
        if name is None and isinstance(function, dict):
            name = function.get("name")
        arguments = getattr(function, "arguments", None) if function is not None else None
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments")

        parsed_arguments = arguments
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"_raw_arguments": arguments}
        if parsed_arguments is None:
            parsed_arguments = {}

        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": str(name or ""),
                    "arguments": parsed_arguments,
                },
            }
        )
    return normalized


def strip_tool_call_block(response: Any) -> str:
    response = str(response or "")
    response = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", response, flags=re.DOTALL)
    return response.strip()


def build_structured_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": str(name or ""),
            "arguments": arguments or {},
        },
    }


def build_assistant_tool_message(
    *,
    response_text: str,
    structured_tool_call: dict[str, Any] | None,
) -> dict[str, Any]:
    content = extract_think_block(response_text)
    if not content:
        content = strip_tool_call_block(response_text)
    return {
        "role": "assistant",
        "content": str(content or ""),
        "tool_calls": [structured_tool_call] if structured_tool_call else None,
    }
