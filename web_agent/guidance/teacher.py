"""Score original action contents using references and actual interaction materials."""

from dataclasses import dataclass
import json

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from web_agent.guidance.feedback import SERVICE_ERRORS
from web_agent.guidance.records import fact_key, material_in_observation, unpack
from web_agent.skills import Skill


@dataclass(frozen=True)
class TeacherContext:
    """Complete teacher prompt IDs, including the assistant generation prefix.

    ``prefix_ids`` replaces the ordinary prompt; it is not an additional prefix
    to concatenate with that prompt. Response IDs are appended separately.
    """

    context: str
    reason: str
    used_facts: list[dict]
    used_skills: tuple[Skill, ...]
    prefix_ids: tuple[int, ...]


def validate_execution_feedback(feedback):
    """Require feedback collected from this action, without reconstructing history."""
    if "execution_feedback" not in feedback:
        raise ValueError("Every action must carry execution_feedback from collection")
    execution = feedback["execution_feedback"]
    if (not isinstance(execution, dict)
            or set(execution) != {"turn", "status", "text"}
            or type(execution["turn"]) is not int
            or execution["turn"] != feedback["turn"]
            or not isinstance(execution["status"], str) or not execution["status"]
            or not isinstance(execution["text"], str)):
        raise ValueError("execution_feedback must contain the action turn, status and text")
    return execution


def _guidance_execution_reason(feedback):
    task, kind = feedback["task"], feedback["action_kind"]
    if task != "searchqa":
        raise ValueError(f"Unsupported guidance task: {task}")
    execution = validate_execution_feedback(feedback)
    if feedback["status"] in SERVICE_ERRORS or execution["status"] in SERVICE_ERRORS:
        return "service_error"
    if "action_executed" not in feedback:
        return "missing_action_execution"
    if feedback["action_executed"] is not True:
        return "action_not_executed"
    if kind not in ("search", "answer"):
        return "unknown_action"
    if kind == "search":
        if (execution["status"] not in ("success", "no_results")
                or execution["status"] != feedback["status"]):
            return "untrusted_execution_feedback"
    elif execution["status"] != "executed":
        return "untrusted_execution_feedback"
    return "ready"


def _fixed_guidance_lines(feedback):
    execution_status = {"turn": feedback["turn"],
                        "status": feedback["execution_feedback"]["status"],
                        "action_executed": feedback["action_executed"]}
    reference = feedback.get("reference") or {}
    lines = ["[Task reference]", json.dumps(
        {"answer_aliases": reference.get("answer_aliases", [])}, ensure_ascii=False),
        "[Actual execution state]", json.dumps(execution_status, ensure_ascii=False)]
    original_state = {"turn": feedback["turn"],
                      "remaining_actions": (feedback.get("skill_state") or {}).get("remaining_actions")}
    return tuple(["[Original action state]", json.dumps(original_state, ensure_ascii=False), *lines])


def _format_skill(skill):
    return "\n".join((f"{skill.skill_id}: {skill.title}", skill.instruction,
                      f"Application conditions: {skill.when_to_apply}"))


def _format_material_record(item):
    return (
        f"{item['timing']}; first turn {item['first_turn']}; observed turn {item['observed_turn']}; "
        f"first visible turn {item['first_visible_turn']}; repeat {item['is_repeat']}; "
        f"{item['source']}; source ID {item['source_id']}; {item['position']}; "
        f"relation {item['relation']}"
        + (f"; value {item['value']}" if "value" in item else "")
    )


def _format_guidance_context(fixed_lines, fact_lines, skill_blocks):
    lines = [*fixed_lines, "[Actual materials and their timing]", *fact_lines]
    if skill_blocks:
        lines.extend(("[Candidate action knowledge]",
                      "Apply a principle only when its stated application conditions match the "
                      "original task, action and prior interaction history. A retrieved principle "
                      "may be inapplicable. Keep the task requirements and requested response format. "
                      "Do not require additional searches, subquestions, comparisons or answer "
                      "formats that the task does not call for. These principles do not establish "
                      "execution materials.",
                      *skill_blocks))
    lines.append("[Original interaction history follows]")
    return "\n".join(lines) + "\n"


# Teacher continuation instruction for SearchQA.
_TEACHER_INSTRUCTION = (
    "Continue the current action from the given original response prefix. "
    "Use the task reference, actual results and applicable knowledge. Preserve "
    "the prefix and the required action format. Provide the continuation directly, "
    "without assessment, commentary or restarting the response. Treat results "
    "obtained after the original action as feedback, not as observations available "
    "beforehand. Choose only operations available in the original interaction "
    "state. References and principles are guidance, not execution records."
)


def _render_chat(tokenizer, messages, chat_template_kwargs):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs,
    )


def _visible_original_messages(tokenizer, original_messages, prompt_ids, chat_template_kwargs):
    """Read original message content without restoring discarded prompt tokens.

    ``raw_prompt`` can retain history lost during left truncation. Verify the
    actual token suffix against the full original template before reading it.
    If truncated, remove all characters covered by discarded tokens, including
    characters split across retained and discarded byte tokens.
    """
    if original_messages is None:
        return (), "missing_original_messages"
    messages = list(original_messages)
    if not messages or any(
        not isinstance(message, dict)
        or message.get("role") not in ("system", "user", "assistant", "tool")
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("Original messages must contain roles and text content")
    messages = [dict(message) for message in messages]
    rendered = _render_chat(tokenizer, messages, chat_template_kwargs)
    full_ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    actual_values = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else prompt_ids
    actual_ids = tuple(int(token_id) for token_id in actual_values)
    if (not actual_ids or len(actual_ids) > len(full_ids)
            or full_ids[-len(actual_ids):] != actual_ids):
        return (), "original_prompt_mismatch"
    removed = len(full_ids) - len(actual_ids)
    hidden_end = 0
    if removed:
        try:
            encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        except (NotImplementedError, TypeError, ValueError):
            return (), "visible_prompt_unavailable"
        offsets = encoded.get("offset_mapping")
        if (tuple(encoded["input_ids"]) != full_ids or offsets is None
                or len(offsets) != len(full_ids)
                or any(not 0 <= start <= end <= len(rendered) for start, end in offsets)):
            return (), "visible_prompt_unavailable"
        hidden_end = max((end for start, end in offsets[:removed]), default=0)

    visible = []
    for index, message in enumerate(messages):
        # A marker locates the rendered content even when identical text occurs
        # in multiple messages. Template wrappers are never nested in a user turn.
        marker = f"__WEB_AGENT_ORIGINAL_MESSAGE_{index}__"
        while marker in rendered:
            marker += "_"
        marked_messages = [dict(item) for item in messages]
        marked_messages[index]["content"] = marker
        marked = _render_chat(tokenizer, marked_messages, chat_template_kwargs)
        if marked.count(marker) != 1:
            return (), "visible_prompt_unavailable"
        before, after = marked.split(marker)
        if (not rendered.startswith(before) or not rendered.endswith(after)
                or len(before) + len(after) > len(rendered)):
            return (), "visible_prompt_unavailable"
        start, end = max(len(before), hidden_end), len(rendered) - len(after)
        if start < end:
            visible.append({"role": message["role"], "content": rendered[start:end]})
    if not visible:
        return (), "visible_prompt_unavailable"
    return tuple(visible), "ready"


def _visible_candidates(feedback, facts):
    turn = feedback["turn"]
    def record_id(item):
        return fact_key(item), item["observed_turn"], item["text"]

    prior_required = feedback["action_kind"] == "answer"
    recorded = feedback["before"] if prior_required else feedback["after"]
    recorded_ids = {record_id(item) for item in recorded}
    candidates = []
    for item in facts:
        if (not item["student_visible"] or item["observed_turn"] > turn
                or record_id(item) not in recorded_ids):
            continue
        if prior_required and item["observed_turn"] >= turn:
            continue
        if not prior_required and item["observed_turn"] != turn:
            continue
        candidate = {**item, "timing": "before action" if item["observed_turn"] < turn else "after action"}
        candidates.append(candidate)
    return candidates


def fit_guidance_context(feedback, facts, skills, tokenizer, prompt_ids, max_prompt_length,
                         *, original_messages=None, chat_template_kwargs=None):
    """Retain actual materials before allocating space to action knowledge.

    Prior materials must be fully present in the ordinary prompt. Their source
    records remain in the prefix without repeating that original text. Newly
    obtained materials are kept in full when absent from the ordinary prompt.
    Scoring retains the visible original message content and every original
    response prefix. Returned IDs include the complete teacher chat template.
    """
    feedback = unpack(feedback)
    execution = validate_execution_feedback(feedback)
    skills = tuple(skills)
    if len(skills) > 3 or any(not isinstance(skill, Skill) for skill in skills):
        raise ValueError("Teacher input accepts at most three existing skills")
    if len({(skill.task, skill.skill_id) for skill in skills}) != len(skills):
        raise ValueError("Teacher skills must have unique source identities")
    if any(skill.task != feedback["task"] or feedback["action_kind"] not in skill.actions for skill in skills):
        raise ValueError("Teacher skill does not apply to this task and action")
    if len(prompt_ids) >= max_prompt_length:
        return TeacherContext("", "prompt_at_budget", [], (), ())
    reason = _guidance_execution_reason(feedback)
    if reason != "ready":
        return TeacherContext("", reason, [], (), ())

    facts = list(facts)
    if not facts:
        return TeacherContext("", "missing_required_materials", [], (), ())
    candidates = _visible_candidates(feedback, facts)
    if (len(candidates) != len(facts)
            or any(not isinstance(item["text"], str) or not item["text"].strip()
                   for item in candidates)):
        return TeacherContext("", "required_materials_unavailable", [], (), ())

    ordinary_prompt = tokenizer.decode(
        prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    template_kwargs = dict(chat_template_kwargs or {})
    if {"tokenize", "add_generation_prompt", "return_tensors", "return_dict"} & template_kwargs.keys():
        raise ValueError("Chat template options must not override prompt rendering")
    original_visible, reason = _visible_original_messages(
        tokenizer, original_messages, prompt_ids, template_kwargs,
    )
    if reason != "ready":
        return TeacherContext("", reason, [], (), ())
    for item in candidates:
        item["in_original_prompt"] = material_in_observation(
            item["text"], ordinary_prompt, feedback["task"],
        )
        if item["observed_turn"] < feedback["turn"]:
            if not item["in_original_prompt"]:
                return TeacherContext("", "prior_material_not_in_prompt", [], (), ())
        elif not material_in_observation(item["text"], execution["text"], feedback["task"]):
            return TeacherContext("", "material_not_in_execution", [], (), ())

    fixed_lines = _fixed_guidance_lines(feedback)

    def encode_context(fact_lines, selected_skill_blocks):
        guidance = _format_guidance_context(fixed_lines, fact_lines, selected_skill_blocks)
        history = "\n".join(
            f"[Original {message['role']} message]\n{message['content']}"
            for message in original_visible
        )
        instruction = _TEACHER_INSTRUCTION
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": guidance + history}]
        text = _render_chat(tokenizer, messages, template_kwargs)
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        return text, ids

    fact_lines = []
    # Reuse ordinary prompt text while retaining every material identity.
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["text"], []).append(candidate)
    for body, records in groups.items():
        body_line = ("Full original material is present in the original interaction history."
                     if all(item["in_original_prompt"] for item in records) else body)
        fact_lines.extend((body_line, *map(_format_material_record, records)))
    retained_skills, retained_skill_blocks = (), ()
    context, prefix_ids = encode_context(fact_lines, retained_skill_blocks)
    if len(prefix_ids) > max_prompt_length:
        return TeacherContext("", "required_context_over_budget", [], (), ())
    for skill in skills:
        skill_block = _format_skill(skill)
        candidate_context, candidate_ids = encode_context(
            fact_lines, retained_skill_blocks + (skill_block,),
        )
        if len(candidate_ids) <= max_prompt_length:
            retained_skills += (skill,)
            retained_skill_blocks += (skill_block,)
            context, prefix_ids = candidate_context, candidate_ids
    return TeacherContext(context, "ready", candidates, retained_skills, prefix_ids)


def build_teacher_batch(batch, tokenizer, max_prompt_length):
    """Pad complete teacher prompts and append the unchanged response IDs."""
    response_length = batch.batch["responses"].size(1)
    if response_length <= 0 or batch.batch["input_ids"].size(1) <= response_length:
        raise ValueError("Teacher scoring requires the original prompt and response")
    if batch.batch["attention_mask"].shape != batch.batch["input_ids"].shape:
        raise ValueError("Teacher attention must match the original input shape")
    if not torch.equal(batch.batch["input_ids"][:, -response_length:], batch.batch["responses"]):
        raise ValueError("Teacher scoring must preserve the original response IDs")
    rows, masks = [], []
    prefix_rows = batch.non_tensor_batch["teacher_context_ids"]
    if len(prefix_rows) != batch.batch.batch_size[0]:
        raise ValueError("Teacher context IDs must correspond to every input row")
    for index, prefix in enumerate(prefix_rows):
        if (not isinstance(prefix, tuple) or not prefix
                or any(type(token_id) is not int or token_id < 0 for token_id in prefix)):
            raise ValueError("Scored teacher context IDs must be a nonempty tuple of token integers")
        attention = batch.batch["attention_mask"][index]
        if len(prefix) > max_prompt_length:
            raise ValueError("Teacher context exceeded its allocated prompt space")
        valid_ids = torch.tensor(prefix, dtype=torch.long)
        pad = max_prompt_length - len(valid_ids)
        full = torch.cat((torch.full((pad,), tokenizer.pad_token_id, dtype=torch.long),
                          valid_ids, batch.batch["responses"][index].cpu().long()))
        mask = torch.cat((torch.zeros(pad, dtype=torch.long), torch.ones(len(valid_ids), dtype=torch.long),
                          attention[-response_length:].cpu().long()))
        rows.append(full)
        masks.append(mask)
    attention_mask = torch.stack(masks)
    return DataProto.from_dict(tensors={
        "input_ids": torch.stack(rows), "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask), "responses": batch.batch["responses"],
    })
