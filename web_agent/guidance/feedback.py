import re

from web_agent.guidance.records import (
    MaterialSelection, material_in_observation, unpack,
)


SERVICE_ERRORS = ("api_error", "processing_error", "unknown_api_state")
MATERIAL_RELATIONS = {"searchqa": "retrieved_material"}
MATERIAL_SOURCES = {"searchqa": "search_document"}


def _visible_at(item, turn, before):
    if item.get("student_visible") is not True or not str(item.get("text", "")).strip():
        return False
    observed, first_visible = item.get("observed_turn"), item.get("first_visible_turn")
    if not isinstance(observed, int) or not isinstance(first_visible, int):
        return False
    return ((observed < turn and first_visible <= observed) if before else
            (observed == turn and first_visible <= turn))


def _actual_material(item, task):
    return (item.get("source_verified") is True
            and item.get("source") == MATERIAL_SOURCES[task]
            and item.get("relation") == MATERIAL_RELATIONS[task])


def returned_visible_materials(feedback):
    """Collect complete visible materials from the current action return."""
    task = feedback["task"]
    return [item for item in feedback.get("after", [])
            if _visible_at(item, feedback["turn"], before=False) and _actual_material(item, task)]


def action_kind(command, task):
    if task != "searchqa":
        raise ValueError(f"Unsupported guidance task: {task}")
    lower = str(command).strip().casefold()
    if re.fullmatch(r"<search>.+</search>", lower, re.DOTALL):
        return "search"
    return "answer" if re.fullmatch(r"<answer>.+</answer>", lower, re.DOTALL) else "invalid"


def select_action_materials(feedback, command, valid=True):
    """Select actual materials to accompany this original action for scoring."""
    feedback = unpack(feedback)
    task, turn = feedback["task"], feedback["turn"]
    kind = action_kind(command, task)
    if feedback.get("status") in SERVICE_ERRORS or feedback.get("service_error"):
        return MaterialSelection("service_error", [])
    if not valid or feedback.get("is_action_valid") is False or kind == "invalid":
        return MaterialSelection("invalid_action", [])
    if "action_executed" not in feedback:
        return MaterialSelection("missing_action_execution", [])
    if feedback["action_executed"] is not True:
        return MaterialSelection("action_not_executed", [])
    actual = feedback.get("projected_action")
    if actual is not None and str(actual).strip() != str(command).strip():
        return MaterialSelection("action_record_mismatch", [])
    execution = feedback.get("execution_feedback")
    if not isinstance(execution, dict) or execution.get("turn") != turn:
        return MaterialSelection("missing_actual_execution_feedback", [])
    if execution.get("status") in SERVICE_ERRORS:
        return MaterialSelection("service_error", [])
    if execution.get("status") not in ("success", "no_results", "executed"):
        return MaterialSelection("unreliable_execution_feedback", [])

    uses_prior = kind == "answer"
    if uses_prior:
        state = feedback.get("skill_state") or {}
        observation = state.get("observation_text")
        if not isinstance(observation, str) or not observation.strip():
            return MaterialSelection("missing_original_observation", [])
        materials = [item for item in feedback.get("before", [])
                     if _visible_at(item, turn, before=True) and _actual_material(item, task)
                     and material_in_observation(item["text"], observation, task)]
        if materials:
            return MaterialSelection("prior_seen_materials", materials)
        return MaterialSelection("no_current_prior_materials", [])

    if task == "searchqa" and kind == "search" and execution.get("status") != "success":
        return MaterialSelection("no_returned_materials", [])
    materials = [item for item in returned_visible_materials(feedback)
                 if material_in_observation(item["text"], execution.get("text"), task)]
    if materials:
        return MaterialSelection("actual_returned_materials", materials)
    return MaterialSelection("no_returned_materials", [])
