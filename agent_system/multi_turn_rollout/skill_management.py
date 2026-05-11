from __future__ import annotations

from typing import Any

from agent_system.alfworld_task_utils import (
    infer_alfworld_skill_category,
    infer_alfworld_task_family_from_gamefile,
)


FINAL_SKILL_MANAGEMENT_LINE = (
    "Now output only the final skill-management decision using the required "
    "<tool_call> JSON format."
)


def env_name_from_config(config: Any) -> str:
    env_cfg = getattr(config, "env", None)
    if env_cfg is None:
        return ""
    env_name = getattr(env_cfg, "env_name", None)
    if env_name is None and hasattr(env_cfg, "get"):
        env_name = env_cfg.get("env_name", "")
    return str(env_name or "").lower()


def is_alfworld_env(config: Any) -> bool:
    env_name = env_name_from_config(config)
    return not env_name or "alfworld" in env_name


def is_webshop_env(config: Any) -> bool:
    return "webshop" in env_name_from_config(config)


def is_search_env(config: Any) -> bool:
    return env_name_from_config(config) == "search"


def get_alfworld_gamefile(envs: Any, item: int) -> str | None:
    gamefiles = getattr(envs, "gamefile", None)
    if isinstance(gamefiles, (list, tuple)) and item < len(gamefiles):
        value = gamefiles[item]
        return str(value) if value is not None else None
    return None


def get_webshop_goal_idx(envs: Any, item: int) -> int | None:
    goal_idxs = getattr(envs, "goal_idx", None)
    if isinstance(goal_idxs, (list, tuple)) and item < len(goal_idxs):
        value = goal_idxs[item]
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def get_webshop_goal(envs: Any, item: int) -> str | None:
    goals = getattr(envs, "goals", None)
    if isinstance(goals, (list, tuple)) and item < len(goals):
        value = goals[item]
        return str(value) if value is not None else None
    return None


def infer_alfworld_task_category(task: str, gamefile: str | None = None) -> str:
    return infer_alfworld_skill_category(task, gamefile=gamefile)


def infer_alfworld_success_family(task: str, gamefile: str | None = None) -> str:
    family = infer_alfworld_task_family_from_gamefile(gamefile)
    if family:
        return family
    category = infer_alfworld_skill_category(task)
    return {
        "heat": "pick_heat_then_place_in_recep",
        "cool": "pick_cool_then_place_in_recep",
        "clean": "pick_clean_then_place_in_recep",
        "look_at_obj_in_light": "look_at_obj_in_light",
        "pick_and_place": "pick_and_place",
    }.get(category, category)


def infer_webshop_skill_category(task: str) -> str:
    """Mirror SkillsOnlyMemory WebShop template routing for skill-bank writes."""
    goal = str(task or "").lower()
    if any(kw in goal for kw in [
        "shirt", "dress", "jacket", "pant", "coat", "sweater",
        "blouse", "clothing", "clothes", "t-shirt",
    ]):
        return "apparel"
    if any(kw in goal for kw in [
        "shoe", "boot", "sneaker", "sandal", "heel", "slipper",
        "footwear",
    ]):
        return "footwear"
    if any(kw in goal for kw in [
        "laptop", "phone", "computer", "tablet", "charger",
        "cable", "headphone", "speaker", "camera", "electronic",
    ]):
        return "electronics"
    if any(kw in goal for kw in [
        "necklace", "ring", "bracelet", "earring", "watch",
        "jewelry", "bag", "purse", "wallet",
    ]):
        return "accessories"
    if any(kw in goal for kw in [
        "furniture", "lamp", "curtain", "pillow", "bedding",
        "decor", "candle", "vase", "rug",
    ]):
        return "home_decor"
    if any(kw in goal for kw in [
        "cream", "lotion", "shampoo", "conditioner", "moisturizer",
        "serum", "makeup", "beauty", "vitamin", "supplement",
    ]):
        return "beauty_health"
    return "other"


def infer_searchqa_query_type(question: str) -> str:
    """Infer SearchQA query type category from question text."""
    text = " ".join(str(question or "").strip().lower().split())
    if not text:
        return "direct_retrieval"

    comparison_cues = [
        " in common", " both ", " same ", " compared to ",
        " compare ", " or the ", " earlier ", " later ",
        " older ", " younger ", " bigger ", " smaller ",
        " longer ", " shorter ", " higher ", " lower ",
        " more than ", " less than ", " before ", " after ",
        " available first", " came first", " happened first",
    ]
    if any(cue in f" {text} " for cue in comparison_cues):
        return "comparison"

    multi_hop_cues = [
        "featuring", "based on", "founded both",
        "part of the area that", "plays in which conference",
        "that was first performed in what year",
        "that was released in what year",
        "within the", "from the film", "from the tv show",
        "that plays", "that represents",
        "that republican legislator", "what town in",
        "what county is part of", "what side of town",
        "which theatre",
    ]
    if any(cue in text for cue in multi_hop_cues):
        return "multi_hop_reasoning"

    relative_markers = sum(
        text.count(token)
        for token in [" that ", " which ", " who ", " whose ", " where ", " when "]
    )
    if len(text.split()) >= 14 and relative_markers >= 2:
        return "multi_hop_reasoning"
    if len(text.split()) >= 18 and relative_markers >= 1:
        return "multi_hop_reasoning"

    entity_attribute_starts = (
        "who ", "where ", "when ", "what year ",
        "what county ", "what conference ", "what station ",
        "what side ", "which province ", "which country ",
        "which town ",
    )
    if text.startswith(entity_attribute_starts):
        return "entity_attribute_lookup"
    if text.startswith("what is ") and " known as" not in text and " in common" not in text:
        return "entity_attribute_lookup"

    return "direct_retrieval"


def infer_expected_skill_category_for_env(
    *,
    config: Any,
    envs: Any,
    item: int,
    task: str,
) -> str | None:
    if is_alfworld_env(config):
        return infer_alfworld_task_category(task, gamefile=get_alfworld_gamefile(envs, item))
    if is_webshop_env(config):
        return infer_webshop_skill_category(task)
    if is_search_env(config):
        return infer_searchqa_query_type(task)
    return None


def env_specific_dump_fields(
    *,
    config: Any,
    envs: Any,
    item: int,
    task: str,
) -> dict[str, Any]:
    if is_alfworld_env(config):
        gamefile = get_alfworld_gamefile(envs, item)
        return {
            "gamefile": gamefile,
            "category": infer_alfworld_task_category(task, gamefile=gamefile),
        }
    if is_webshop_env(config):
        category = infer_webshop_skill_category(task)
        return {
            "env_name": env_name_from_config(config),
            "category": category,
            "webshop_category": category,
            "goal_idx": get_webshop_goal_idx(envs, item),
            "goal": get_webshop_goal(envs, item),
        }
    if is_search_env(config):
        category = infer_searchqa_query_type(task)
        return {
            "env_name": env_name_from_config(config),
            "category": category,
            "task_category": category,
        }
    return {
        "env_name": env_name_from_config(config),
        "category": "unknown",
    }


def build_skill_management_prompt(
    *,
    config: Any,
    task: str,
    episode_reward: float,
    success_value: bool,
    retrieved_skills_text: str,
    episode_trace: str,
    include_trajectory_recap: bool = True,
    category: str | None = None,
    goal_idx: int | None = None,
    goal: str | None = None,
) -> str:
    if is_webshop_env(config):
        return _build_webshop_skill_management_prompt(
            task=task,
            category=category or infer_webshop_skill_category(task),
            episode_reward=episode_reward,
            success_value=success_value,
            retrieved_skills_text=retrieved_skills_text,
            episode_trace=episode_trace,
            include_trajectory_recap=include_trajectory_recap,
            goal_idx=goal_idx,
            goal=goal,
        )

    if is_search_env(config):
        return _build_searchqa_skill_management_prompt(
            task=task,
            category=category or infer_searchqa_query_type(task),
            episode_reward=episode_reward,
            success_value=success_value,
            retrieved_skills_text=retrieved_skills_text,
            episode_trace=episode_trace,
            include_trajectory_recap=include_trajectory_recap,
        )

    return _build_alfworld_skill_management_prompt(
        task=task,
        category=category or infer_alfworld_task_category(task),
        episode_reward=episode_reward,
        success_value=success_value,
        retrieved_skills_text=retrieved_skills_text,
        episode_trace=episode_trace,
        include_trajectory_recap=include_trajectory_recap,
    )


def _build_alfworld_skill_management_prompt(
    *,
    task: str,
    category: str,
    episode_reward: float,
    success_value: bool,
    retrieved_skills_text: str,
    episode_trace: str,
    include_trajectory_recap: bool,
) -> str:
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
    _append_trajectory_recap(lines, episode_trace, include_trajectory_recap)
    return "\n".join(lines)


def _build_webshop_skill_management_prompt(
    *,
    task: str,
    category: str,
    episode_reward: float,
    success_value: bool,
    retrieved_skills_text: str,
    episode_trace: str,
    include_trajectory_recap: bool,
    goal_idx: int | None,
    goal: str | None,
) -> str:
    outcome = "Success" if success_value else "Failure"

    lines = [
        "You are reviewing a completed WebShop episode.",
        "Decide whether the skill bank should be updated based on reusable evidence from this shopping episode.",
        "",
        "Rules:",
        "- Call at most one skill-management tool.",
        "- Choose the tool that best reflects whether the episode reveals a reusable shopping lesson that is missing or incorrect in the current bank.",
        "- Use propose_skill only for a genuinely new reusable lesson.",
        "- Use update_skill only when an existing skill should be revised.",
        "- Use keep_skill only when the current retrieved skills already cover the observed strategy or failure pattern well enough.",
        "- Base your decision on the shopping task, episode evidence, and the current retrieved skills.",
        "- Every decision must be justified by concrete evidence from THIS EPISODE, not by generic real-world shopping advice.",
        "- Compare the observed pattern against the retrieved skills explicitly; do not choose keep_skill just because some broad WebShop guidance exists.",
        "- Success alone is not a reason to keep_skill: if a successful episode demonstrates a concise reusable strategy that is not already covered, propose_skill or update_skill.",
        "- Failure alone is not a reason to change the bank: if the failure does not reveal a concrete reusable lesson beyond the current retrieved skills, use keep_skill.",
        "- Favor lessons about search query formulation, attribute filtering, option selection, product comparison, and the search-click-option-buy workflow.",
        "- Treat missed required attributes, wrong option selection, premature buying, ineffective repeated searches, and losing track of constraints as strong evidence for propose_skill or update_skill unless an existing retrieved skill already states that rule explicitly.",
        "- For keep_skill, the reason must name the specific observed pattern from this episode and explain why the current retrieved skills already cover it OR why the failure does not justify a new reusable skill.",
        "- Do not use a generic keep_skill reason. Tie it to the actual trajectory, such as a bad first click, an over-specific query, repeated option clicks on the wrong candidate, or a premature buy.",
        "- Do not propose or update a verification skill unless the trajectory clearly shows that missing verification, if performed, would likely have changed the decision and fixed the outcome.",
        "- Do not default to 'verify more attributes', 'check description/features', or 'confirm product type before buying/selecting options' unless the trajectory evidence specifically supports that lesson better than a search, result-screening, or visible-option lesson.",
        "- If the failure is mainly caused by noisy search results, a bad first click, or an overloaded query, prefer a search-query or result-screening lesson instead of a product-page verification lesson.",
        "- Add or revise a skill only if it is generic, concise, and useful for future WebShop tasks.",
        "- Do not propose or update a skill that merely restates an already retrieved skill with minor wording changes.",
        "- Only write skills grounded in WebShop shopping behavior, product attributes, search results, item pages, options, and buying decisions.",
        "- Do not store one-off product titles, product IDs, page indices, exact prices, exact search result positions, or episode narration unless the lesson is clearly reusable.",
        "- Do not include meta-commentary about skill-bank decisions, prompt quality, guidance quality, success/failure labels, or whether the bank should change.",
        "- Do not output placeholders, ellipses, half-finished text, or copied trajectory fragments.",
        "- For propose_skill or update_skill, make the title short and canonical, and make principle/when_to_apply concise reusable rules rather than episode summaries.",
        "- For update_skill, set skill_id to the exact retrieved skill title if no explicit numeric skill_id is shown.",
        "- Do not output a WebShop <action>; the episode is already over.",
        "- Do not choose environment actions such as search[...], click[...], buy, back to search, next, previous, or option clicks.",
        "- First reason inside <think> </think>, then output exactly one skill-management tool call in JSON.",
        "- Keep <think> extremely short: 1-3 sentences, no bullet points, no long episode recap, and no copied trajectory details.",
        "- The JSON must be enclosed in <tool_call> </tool_call> tags and must be the final content.",
        "- For task-specific skills, set category to the WebShop Category shown below.",
        "- Use category=\"general\" only if the lesson is clearly reusable across multiple product categories.",
        "",
        "Output requirements:",
        "- Reason inside <think>...</think>, but keep it brief and decision-focused.",
        "- Your final content must be exactly one skill-management tool call wrapped in <tool_call>...</tool_call>.",
        "- Inside <tool_call>...</tool_call>, output only valid JSON with keys 'name' and 'arguments'.",
        "- If you propose or update a skill, output compact reusable text, not long paragraphs.",
        "- If you choose keep_skill, the reason must reference the actual failure/success pattern from this episode rather than a generic statement about the benchmark.",
        "- Copy argument field names exactly, including required fields such as evidence for propose_skill.",
        "- Do not use markdown code fences.",
        "",
        "Valid output formats:",
        '<think>...</think>\n<tool_call>{"name":"keep_skill","arguments":{"reason":"..."}}</tool_call>',
        f'<think>...</think>\n<tool_call>{{"name":"propose_skill","arguments":{{"category":"{category}","title":"...","principle":"...","when_to_apply":"...","evidence":"..."}}}}</tool_call>',
        '<think>...</think>\n<tool_call>{"name":"update_skill","arguments":{"skill_id":"...","title":"...","principle":"...","when_to_apply":"...","reason":"..."}}</tool_call>',
        "",
        f"Shopping Task: {task}",
        f"WebShop Category: {category}",
        f"Episode Outcome: {outcome}",
        f"Episode Reward: {episode_reward}",
    ]
    if goal_idx is not None:
        lines.append(f"Goal Index: {goal_idx}")
    if goal:
        lines.append(f"Goal Text: {goal}")
    lines.extend([
        "",
        "Retrieved Skills:",
        str(retrieved_skills_text or "No retrieved skills were available."),
        "",
    ])
    _append_trajectory_recap(lines, episode_trace, include_trajectory_recap)
    return "\n".join(lines)


def _build_searchqa_skill_management_prompt(
    *,
    task: str,
    category: str,
    episode_reward: float,
    success_value: bool,
    retrieved_skills_text: str,
    episode_trace: str,
    include_trajectory_recap: bool,
) -> str:
    outcome = "Success" if success_value else "Failure"

    lines = [
        "You are reviewing a completed SearchQA episode.",
        "Decide whether the skill bank should be updated based on reusable evidence from this question-answering episode.",
        "",
        "Rules:",
        "- Call at most one skill-management tool.",
        "- Choose the tool that best reflects whether the episode reveals a reusable search-and-answer lesson that is missing or incorrect in the current bank.",
        "- Use propose_skill only for a genuinely new reusable lesson.",
        "- Use update_skill only when an existing skill should be revised.",
        "- Use keep_skill only when the current retrieved skills already cover the observed reasoning/search pattern well enough.",
        "- Base your decision on the question, episode evidence, and the current retrieved skills.",
        "- Every decision must be justified by concrete evidence from THIS EPISODE, not by generic advice about question answering.",
        "- Compare the observed pattern against the retrieved skills explicitly; do not choose keep_skill just because some broad search advice already exists.",
        "- Success alone is not a reason to keep_skill: if a successful episode demonstrates a concise reusable search strategy that is not already covered, propose_skill or update_skill.",
        "- Failure alone is not a reason to change the bank: if the failure does not reveal a concrete reusable lesson beyond the current retrieved skills, use keep_skill.",
        "- Prefer lessons about decomposing the question, crafting precise queries, refining weak queries, resolving ambiguity, deciding when evidence is sufficient, and cross-checking critical facts before answering.",
        "- Treat repeated vague queries, repeated unproductive search without refinement, unsupported answers, missed ambiguity, and failure to decompose multi-hop questions as strong evidence for propose_skill or update_skill unless an existing retrieved skill already states that rule explicitly.",
        "- Do not propose or update a skill if the episode only reflects a one-off fact, a narrow named entity, an exact date, or a generic lesson already stated clearly in the retrieved skills.",
        "- Add or revise a skill only if it is generic, concise, and useful for future SearchQA episodes.",
        "- Do not propose or update a skill that merely restates an already retrieved skill with minor wording changes.",
        "- Only write skills grounded in search-based question answering, evidence gathering, query refinement, and final answer decisions.",
        "- Do not store one-off entities, exact dates, exact search results, or copied snippets unless the lesson is clearly reusable.",
        "- Do not include meta-commentary about skill-bank decisions, prompt quality, success/failure labels, or whether the bank should change.",
        "- Do not output placeholders, ellipses, half-finished text, or copied trajectory fragments.",
        "- For propose_skill or update_skill, make the title short and canonical, and make principle/when_to_apply concise reusable rules rather than episode summaries.",
        "- For update_skill, set skill_id to the exact retrieved skill title if no explicit numeric skill_id is shown.",
        "- Do not output another <search> query or an <answer>; the episode is already over.",
        "- First reason inside <think> </think>, then output exactly one skill-management tool call in JSON.",
        "- Keep <think> extremely short: 1-3 sentences, no bullet points, no long episode recap, and no copied trajectory details.",
        "- The JSON must be enclosed in <tool_call> </tool_call> tags and must be the final content.",
        "- For task-specific skills, set category to the SearchQA Query Type shown below.",
        "- Use category=\"general\" only if the lesson is clearly reusable across many question types.",
        "",
        "Output requirements:",
        "- Reason inside <think>...</think>, but keep it brief and decision-focused.",
        "- You MUST output exactly two blocks in this order:",
        "1. one <think>...</think> block",
        "2. one <tool_call>...</tool_call> block",
        "- Your response is invalid if:",
        "- it does not begin with <think>",
        "- it does not contain exactly one <think>...</think> block",
        "- it does not contain exactly one <tool_call>...</tool_call> block",
        "- any text appears before <think>, between </think> and <tool_call> other than whitespace, or after </tool_call>",
        "- Inside <tool_call>...</tool_call>, output only valid JSON with keys \"name\" and \"arguments\".",
        "- If you propose or update a skill, output compact reusable text, not long paragraphs.",
        "- If you choose keep_skill, the reason must reference the actual success/failure pattern from this episode rather than a generic statement.",
        "- Copy argument field names exactly, including required fields such as evidence for propose_skill.",
        "- Do not use markdown code fences.",
        "",
        "Valid output formats:",
        '<think>...</think>\n<tool_call>{"name":"keep_skill","arguments":{"reason":"..."}}</tool_call>',
        f'<think>...</think>\n<tool_call>{{"name":"propose_skill","arguments":{{"category":"{category}","title":"...","principle":"...","when_to_apply":"...","evidence":"..."}}}}</tool_call>',
        '<think>...</think>\n<tool_call>{"name":"update_skill","arguments":{"skill_id":"...","title":"...","principle":"...","when_to_apply":"...","reason":"..."}}</tool_call>',
        "",
        f"Question: {task}",
        f"SearchQA Query Type: {category}",
        f"Episode Outcome: {outcome}",
        f"Episode Reward: {episode_reward}",
        "",
        "Retrieved Skills:",
        str(retrieved_skills_text or "No retrieved skills were available."),
        "",
    ]
    _append_trajectory_recap(lines, episode_trace, include_trajectory_recap)
    return "\n".join(lines)


def _append_trajectory_recap(lines: list[str], episode_trace: str, include_trajectory_recap: bool) -> None:
    if include_trajectory_recap:
        lines.extend([
            "Episode Trajectory:",
            episode_trace or "No episode trajectory was recorded.",
            "END EPISODE TRAJECTORY.",
            "",
        ])
    lines.append(FINAL_SKILL_MANAGEMENT_LINE)
