"""Collect the observation associated with each environment action."""

from copy import deepcopy
import json
import re

import numpy as np

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from web_agent.guidance.feedback import action_kind
from web_agent.guidance.records import annotate_occurrences, append_new, merge_records, pack, unpack
from web_agent.guidance.searchqa import search_facts, search_reference


def collect_execution_feedback(task, kind, turn, info, anchor, action_executed):
    """Keep this action's visible return and explicit execution status."""
    record = {"turn": int(turn), "status": "action_not_executed", "text": ""}
    if not bool(info["is_action_valid"]):
        record["status"] = "invalid_action"
        return record
    if not action_executed:
        return record
    if kind == "answer":
        record["status"] = "executed"
        return record
    if task != "searchqa":
        raise ValueError(f"Unsupported execution feedback task: {task}")
    status = info["retrieval_status"]
    record["status"] = status
    match = (re.fullmatch(r"\s*<information>(.*)</information>\s*", anchor, re.DOTALL)
             if isinstance(anchor, str) else None)
    try:
        payload = json.loads(match.group(1)) if match is not None else None
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        record["text"] = payload["result"]
    elif status in ("success", "no_results"):
        record["status"] = "processing_error"
    return record


class ObservationTrajectoryCollector(TrajectoryCollector):

    def __init__(self, config, tokenizer, processor=None):
        super().__init__(config, tokenizer, processor)
        self._rollout_env_kwargs = None
        self._collect_guidance = False
        self._training_turn = 0

    def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):

        self._training_turn = 0
        if self._collect_guidance and self.config.algorithm.obs_distill.task == "searchqa":
            self._rollout_env_kwargs = gen_batch.non_tensor_batch["env_kwargs"]
        elif "env_kwargs" in gen_batch.non_tensor_batch:
            self._rollout_env_kwargs = gen_batch.non_tensor_batch["env_kwargs"]
        else:
            self._rollout_env_kwargs = None

        return super().vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        rollout_batch = type(gen_batch)(
            batch=gen_batch.batch,
            non_tensor_batch=deepcopy(gen_batch.non_tensor_batch),
            meta_info=dict(gen_batch.meta_info),
        )
        self._collect_guidance = bool(is_train)
        self._training_turn = 0
        self._rollout_env_kwargs = None
        try:
            batch = super().multi_turn_loop(rollout_batch, actor_rollout_wg, envs, is_train=is_train)
        finally:
            self._collect_guidance = False
            self._training_turn = 0
            self._rollout_env_kwargs = None

        if "retrieval_status" not in batch.non_tensor_batch:
            return batch
        failed_rows = np.flatnonzero([
            value in ("api_error", "processing_error", "unknown_api_state")
            for value in batch.non_tensor_batch["retrieval_status"]
        ])
        if not len(failed_rows):
            return batch
        if not is_train:
            raise RuntimeError(
                f"Validation retrieval failed for {len(failed_rows)} action records; "
                "check the retrieval service before scoring this batch"
            )
        uid = batch.non_tensor_batch["uid"]
        failed_groups = {str(uid[index]) for index in failed_rows}
        kept = np.flatnonzero([str(value) not in failed_groups for value in uid])
        if not len(kept):
            raise RuntimeError(
                f"Retrieval failed for all {len(failed_groups)} training groups; "
                "check the configured retrieval service"
            )
        batch = batch.select_idxs(kept)
        batch.meta_info["retrieval_error_groups"] = len(failed_groups)
        return batch


    def preprocess_single_sample(self, item, gen_batch, obs):
        row = super().preprocess_single_sample(item, gen_batch, obs, truncation="left")
        visible_ids = row["input_ids"][row["attention_mask"].bool()]
        row["skill_observation_text"] = self.tokenizer.decode(
            visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        row["observation_anchor"] = row["anchor_obs"]
        row["guidance_seen"] = obs.get("guidance_seen", [pack([])] * len(obs["text"]))[item]
        row["guidance_known"] = obs.get("guidance_known", [pack([])] * len(obs["text"]))[item]
        for key in ("metadata", "reward_model", "extra_info"):
            if key in gen_batch.non_tensor_batch:
                row[key] = gen_batch.non_tensor_batch[key][item]
        if "index" in gen_batch.non_tensor_batch:
            row["task_index"] = int(gen_batch.non_tensor_batch["index"][item])

        if self._rollout_env_kwargs is not None:
            row["env_kwargs"] = self._rollout_env_kwargs[item]

        return row

    def after_env_step(self, batch, observations, next_observations, infos, turn_step):
        if any("retrieval_status" in info for info in infos):
            batch.non_tensor_batch["retrieval_status"] = np.asarray(
                [info.get("retrieval_status", "api_error") for info in infos], dtype=object
            )
        if not self._collect_guidance:
            return
        self._training_turn = int(turn_step) + 1

        task = self.config.algorithm.obs_distill.task
        feedback_rows = []
        next_seen_rows = []
        next_known_rows = []
        for index, info in enumerate(infos):
            seen = unpack(batch.non_tensor_batch["guidance_seen"][index])
            known = unpack(batch.non_tensor_batch["guidance_known"][index])
            metadata = batch.non_tensor_batch.get("metadata", [None] * len(infos))[index]
            env_kwargs = batch.non_tensor_batch["env_kwargs"][index]
            candidates = info.get("retrieval_candidates", [])
            visible_topk = int(self.config.env.search.topk)
            returned = search_facts(candidates, metadata, env_kwargs, turn_step, visible_topk)
            reference = search_reference(metadata, env_kwargs)
            status = info.get("retrieval_status", "not_applicable")

            returned = annotate_occurrences(known, returned)
            projected = str(info["projected_action"])
            kind = action_kind(projected, task)
            action_executed = (kind == "answer" or (
                kind == "search" and bool(info.get("tool_calling", False))
                and status not in ("invalid_query", "not_applicable")
            ))
            execution_feedback = collect_execution_feedback(
                task, kind, turn_step, info,
                next_observations["anchor"][index],
                action_executed,
            )
            task_text = batch.non_tensor_batch["env_kwargs"][index]["question"]
            skill_state = {
                "task": task,
                "task_text": str(task_text),
                "observation_text": str(batch.non_tensor_batch["skill_observation_text"][index]),
                "turn": int(turn_step),
                "remaining_actions": int(self.config.env.max_steps) - int(turn_step),
            }
            record = {
                "skill_state": skill_state,
                "task": task,
                "turn": turn_step,
                "before": seen,
                "after": returned,
                "reference": reference,
                "status": status,
                "action_executed": action_executed,
                "execution_feedback": execution_feedback,
                "is_action_valid": bool(info["is_action_valid"]),
                "service_error": status in ("api_error", "processing_error", "unknown_api_state"),
                "projected_action": projected,
            }
            feedback_rows.append(pack(record))
            next_seen_rows.append(pack(append_new(seen, returned)))
            next_known_rows.append(pack(merge_records(known, returned)))
        batch.non_tensor_batch["guidance_feedback"] = np.asarray(feedback_rows, dtype=object)
        next_observations["guidance_seen"] = next_seen_rows
        next_observations["guidance_known"] = next_known_rows
