from __future__ import annotations

from typing import Optional


ALFWORLD_FAMILY_TO_SKILL_CATEGORY = {
    "pick_and_place": "pick_and_place",
    "pick_two_obj_and_place": "pick_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_clean_then_place_in_recep": "clean",
}


def infer_alfworld_task_family_from_gamefile(gamefile: Optional[str]) -> Optional[str]:
    """Infer the ALFWorld environment task family from a gamefile path."""
    gamefile_l = str(gamefile or "").lower()
    if not gamefile_l:
        return None

    # Check transformed task types before generic pick_and_place.
    for family in (
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
        "pick_and_place",
    ):
        if family in gamefile_l:
            return family
    return None


def map_alfworld_family_to_skill_category(task_family: Optional[str]) -> Optional[str]:
    if task_family is None:
        return None
    return ALFWORLD_FAMILY_TO_SKILL_CATEGORY.get(task_family)


def infer_alfworld_skill_category_from_text(task: str) -> str:
    """Fallback category inference from the natural-language task text."""
    task_l = str(task or "").lower()
    if "look at" in task_l and "under" in task_l:
        return "look_at_obj_in_light"
    if "clean" in task_l or "cleaned" in task_l:
        return "clean"
    if "cool" in task_l or "cooled" in task_l or "cold" in task_l:
        return "cool"
    if "heat" in task_l or "heated" in task_l or "hot" in task_l:
        return "heat"
    return "pick_and_place"


def infer_alfworld_skill_category(task: str, gamefile: Optional[str] = None) -> str:
    """Infer the skill-bank category, preferring ALFWorld metadata when present."""
    task_family = infer_alfworld_task_family_from_gamefile(gamefile)
    skill_category = map_alfworld_family_to_skill_category(task_family)
    if skill_category:
        return skill_category
    return infer_alfworld_skill_category_from_text(task)
