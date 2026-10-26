"""Learn teacher-sampled words on unchanged original response prefixes."""

import math

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from web_agent.training.loop import ActionTrainingLoop

from web_agent.action_mask import ActionDecoder, matches_projected_action
from web_agent.distill import compute_content_weights
from web_agent.guidance.feedback import SERVICE_ERRORS, action_kind, select_action_materials
from web_agent.guidance.records import MaterialSelection, pack, unpack
from web_agent.guidance import teacher, prefix_teacher


class WebAgentTrainer(ActionTrainingLoop):

    def __init__(self, *, skill_bank, **kwargs):
        super().__init__(**kwargs)
        self.skill_bank = skill_bank
        self.action_decoder = ActionDecoder(self.tokenizer)
        method = self.config.algorithm.obs_distill
        self.teacher_context_format = method.teacher_context
        if self.teacher_context_format not in ("messages", "prefix"):
            raise ValueError("teacher_context must be messages or prefix")
        if method.normalization not in ("content_positions", "response_positions"):
            raise ValueError("Unknown content objective normalization")
        self.teacher_prompts = teacher if self.teacher_context_format == "messages" else prefix_teacher


    def _sample_teacher_targets(self, batch):
        """Draw teacher labels before the single ordinary-input scoring pass."""
        num_samples = batch.meta_info.get("num_content_samples", 4)
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_content_samples must be a positive integer")
        active_indices = np.flatnonzero(batch.non_tensor_batch["teacher_score_active"])
        batch.meta_info["teacher_scored_rows"] = int(active_indices.size)
        batch.meta_info["teacher_padding_rows"] = 0
        batch.meta_info["content_scored_rows"] = int(active_indices.size)
        batch.meta_info["content_padding_rows"] = 0
        batch.batch["content_target_ids"] = batch.batch["responses"].detach().unsqueeze(-1).expand(-1, -1, num_samples).clone()
        batch.batch["content_teacher_log_probs"] = torch.zeros_like(batch.batch["content_target_ids"], dtype=torch.float32)
        teacher_log_probs = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        if not active_indices.size:
            return teacher_log_probs
        teacher_inputs = batch.select(
            batch_keys=["input_ids", "attention_mask", "responses", "teacher_score_mask"],
            non_tensor_batch_keys=["teacher_context_ids"],
        )[active_indices]
        teacher_batch = self.teacher_prompts.build_teacher_batch(
            teacher_inputs, self.tokenizer, batch.meta_info["teacher_max_prompt_length"],
        )
        teacher_batch.batch["teacher_score_mask"] = teacher_inputs.batch["teacher_score_mask"].detach().clone()
        teacher_batch.meta_info.update({
            "calculate_entropy": False, "sample_content_targets": True,
            "num_content_samples": num_samples,
        })
        teacher_batch, pad_size = pad_dataproto_to_divisor(teacher_batch, self.actor_rollout_wg.world_size)
        batch.meta_info["teacher_padding_rows"] = pad_size
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_output = unpad_dataproto(teacher_output, pad_size=pad_size)
        indices = torch.as_tensor(active_indices, device=teacher_log_probs.device)
        teacher_log_probs.index_copy_(
            0, indices, teacher_output.batch["old_log_probs"].detach().float().to(teacher_log_probs),
        )
        targets = teacher_output.batch["content_target_ids"].detach()
        expected_shape = (*teacher_inputs.batch["responses"].shape, num_samples)
        if targets.shape != expected_shape or targets.dtype != torch.long:
            raise ValueError("Teacher labels must have the original response positions and a candidate dimension")
        target_destination = batch.batch["content_target_ids"]
        target_destination.index_copy_(0, indices.to(target_destination.device), targets.to(target_destination))
        sampled_q = teacher_output.batch["content_teacher_log_probs"].detach().float()
        if sampled_q.shape != targets.shape:
            raise ValueError("Teacher label probabilities must match all candidate positions")
        batch.batch["content_teacher_log_probs"].index_copy_(
            0, indices.to(batch.batch["content_teacher_log_probs"].device),
            sampled_q.to(batch.batch["content_teacher_log_probs"]),
        )

        return teacher_log_probs

    def _prepare_training_teacher_inputs(self, batch):
        method = self.config.algorithm.obs_distill
        task = method.task
        if task != "searchqa":
            raise ValueError(f"Unsupported guidance task: {task}")
        coef = float(method.guidance_coef)
        beta = float(method.gate_beta)
        num_samples = getattr(method, "num_content_samples", 4)
        if not math.isfinite(coef) or coef < 0:
            raise ValueError("guidance_coef must be finite and nonnegative")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("gate_beta must be finite and positive")
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_content_samples must be a positive integer")
        batch.meta_info.update({
            "teacher_guidance": True, "guidance_coef": coef, "guidance_beta": beta,
            "num_content_samples": num_samples,
            "normalization": method.normalization,
            "teacher_context": self.teacher_context_format,
            "teacher_max_prompt_length": 2 * int(self.config.data.max_prompt_length),
            "training_settings": OmegaConf.to_container(method.training_settings, resolve=True),
        })
        masks, material_available = [], []
        contexts, reasons, material_reasons, used_rows, kinds, parse_states = [], [], [], [], [], []
        selected_skill_rows, used_skill_rows, skill_states = [], [], []
        context_ids, action_texts, action_starts, action_ends = [], [], [], []
        selected_skill_count = used_skill_count = 0
        valid_count = 0
        for index in range(batch.batch.batch_size[0]):
            response_mask = batch.batch["response_mask"][index]
            parsed = self.action_decoder.parse(batch.batch["responses"][index], response_mask, task)
            response = parsed.response
            projected = str(batch.non_tensor_batch["projected_action"][index])
            valid = bool(batch.non_tensor_batch["is_action_valid"][index])
            format_valid = valid
            start, end = parsed.span
            same_action = matches_projected_action(response, projected, task, span=parsed.span)
            action_texts.append(response[start:end])
            action_starts.append(start)
            action_ends.append(end)
            parse_status = (
                "parse_missing" if start == end else "format_invalid" if not format_valid
                else "action_not_executed" if not valid
                else "projection_mismatch" if not same_action else "ready"
            )
            parse_states.append(parse_status)
            training_row = bool(batch.batch["training_row_mask"][index])
            valid_count += training_row and parse_status == "ready"
            feedback = unpack(batch.non_tensor_batch["guidance_feedback"][index])
            self.teacher_prompts.validate_execution_feedback(feedback)
            kind = action_kind(projected, task)
            kinds.append(kind)
            feedback["action_kind"] = kind
            feedback["action_parse_status"] = parse_status
            feedback["format_valid"] = format_valid
            feedback["projection_matches"] = same_action
            turn = int(feedback["turn"])
            if "skill_state" not in feedback:
                raise ValueError("Every action must carry its original Skill selection state")
            skill_state = feedback["skill_state"]
            self.skill_bank.validate_state(skill_state)
            if skill_state["task"] != task or skill_state["turn"] != turn:
                raise ValueError("Skill selection state does not match the original action")
            if skill_state["remaining_actions"] != int(self.config.env.max_steps) - turn:
                raise ValueError("Skill selection state has an incorrect action budget")
            selected_skills = self.skill_bank.select(skill_state, kind, parsed.body_text) if training_row and parse_status == "ready" and kind != "invalid" else ()
            selected_skill_rows.append(pack([skill.record() for skill in selected_skills]))
            selected_skill_count += len(selected_skills)
            skill_states.append(pack(skill_state))
            for phase in ("before", "after"):
                for item in feedback.get(phase, []):
                    observed = int(item["observed_turn"])
                    if int(item["first_turn"]) > observed or (phase == "before" and observed >= turn) or (phase == "after" and observed != turn):
                        raise ValueError("Task fact is outside the action's observation time")
            selection = (select_action_materials(feedback, projected, valid and same_action) if training_row
                         else MaterialSelection("training_padding", []))
            feedback["material_selection_reason"] = selection.reason
            material_reasons.append(selection.reason)

            mask = parsed.body_mask.detach().clone()
            if "loss_mask" in batch.batch:
                mask *= batch.batch["loss_mask"][index, -mask.numel():]
            context, prefix_ids, used_facts, used_skills = "", (), [], ()
            reason = parse_status
            if not training_row:
                reason = "training_padding"
            if reason == "ready":
                if not selection.available:
                    reason = selection.reason
                elif (feedback["status"] in SERVICE_ERRORS
                        or feedback["execution_feedback"]["status"] in SERVICE_ERRORS):
                    reason = "service_error"
                elif "action_executed" not in feedback:
                    reason = "missing_action_execution"
                elif not feedback.get("action_executed", False):
                    reason = "action_not_executed"
                elif not mask.any():
                    reason = "action_content_empty"
                elif kind == "invalid":
                    reason = "unknown_action"
                else:
                    response_width = batch.batch["responses"].size(1)
                    prompt_mask = batch.batch["attention_mask"][index, :-response_width].bool()
                    prompt_ids = batch.batch["input_ids"][index, :-response_width][prompt_mask]
                    facts = selection.facts
                    prompt_options = {}
                    if self.teacher_context_format == "messages":
                        prompt_options = {
                            "original_messages": batch.non_tensor_batch.get(
                                "raw_prompt", [None] * batch.batch.batch_size[0],
                            )[index],
                            "chat_template_kwargs": dict(getattr(self.config.data, "apply_chat_template_kwargs", {})),
                        }
                    fitted = self.teacher_prompts.fit_guidance_context(
                        feedback, facts, selected_skills, self.tokenizer, prompt_ids,
                        batch.meta_info["teacher_max_prompt_length"],
                        **prompt_options,
                    )
                    context, reason = fitted.context, fitted.reason
                    prefix_ids = fitted.prefix_ids
                    used_facts, used_skills = fitted.used_facts, fitted.used_skills
            if reason != "ready":
                mask.zero_()
            material_available.append(selection.available)
            feedback["teacher_score_available"] = bool(mask.any())
            feedback["teacher_score_skip_reason"] = reason
            batch.non_tensor_batch["guidance_feedback"][index] = pack(feedback)
            masks.append(mask)
            contexts.append(context)
            context_ids.append(prefix_ids)
            reasons.append(reason)
            used_rows.append(pack(used_facts))
            used_skill_rows.append(pack([skill.record() for skill in used_skills]))
            used_skill_count += len(used_skills)

        batch.batch["teacher_score_mask"] = torch.stack(masks)
        scored = np.asarray([bool(mask.any()) for mask in masks], dtype=bool)
        prefix_rows = np.empty(len(context_ids), dtype=object)
        for index, prefix_ids in enumerate(context_ids):
            prefix_rows[index] = prefix_ids
        batch.non_tensor_batch.update({
            "teacher_context": np.asarray(contexts, dtype=object),
            "teacher_context_ids": prefix_rows,
            "action_text": np.asarray(action_texts, dtype=object),
            "action_start": np.asarray(action_starts, dtype=np.int64),
            "action_end": np.asarray(action_ends, dtype=np.int64),
            "teacher_skip_reason": np.asarray(reasons, dtype=object),
            "teacher_score_skip_reason": np.asarray(reasons, dtype=object),
            "teacher_material_available": np.asarray(material_available, dtype=bool),
            "teacher_score_active": scored,
            "material_selection_reason": np.asarray(material_reasons, dtype=object),
            "teacher_used_facts": np.asarray(used_rows, dtype=object),
            "teacher_selected_skills": np.asarray(selected_skill_rows, dtype=object),
            "teacher_used_skills": np.asarray(used_skill_rows, dtype=object),
            "teacher_skill_state": np.asarray(skill_states, dtype=object),
            "action_kind": np.asarray(kinds, dtype=object),
            "action_parse_status": np.asarray(parse_states, dtype=object),
        })
        teacher = self._sample_teacher_targets(batch)
        batch.non_tensor_batch.pop("teacher_context_ids")
        score_mask = batch.batch["teacher_score_mask"].bool()
        metrics = {
            "guidance/teacher_scored_rows": batch.meta_info["teacher_scored_rows"],
            "guidance/teacher_scoring_coverage": float(scored.sum() / valid_count) if valid_count else 0.0,
            "guidance/selected_skills": selected_skill_count,
            "guidance/used_skills": used_skill_count,
            "guidance/teacher_padding_rows": batch.meta_info["teacher_padding_rows"],
            "guidance/content_scored_rows": batch.meta_info["content_scored_rows"],
            "guidance/content_padding_rows": batch.meta_info["content_padding_rows"],
            "guidance/teacher_skipped_rows": int((~scored & batch.batch["training_row_mask"].cpu().numpy()).sum()),
            "guidance/training_padding_rows": int((~batch.batch["training_row_mask"]).sum().item()),
            "guidance/valid_actions": valid_count,
            "guidance/material_available_actions": sum(material_available),
            "guidance/material_unscored_actions": int((np.asarray(material_available) & ~scored).sum()),
            "guidance/scored_tokens": int(score_mask.sum().item()),
            "guidance/prompt_budget_actions": sum("budget" in reason for reason in reasons),
            "guidance/retrieval_error_groups": batch.meta_info.get("retrieval_error_groups", 0),
        }
        for kind in ("search", "answer", "invalid"):
            metrics[f"guidance/action_{kind}_count"] = sum(
                bool(real) and value == kind
                for real, value in zip(batch.batch["training_row_mask"], kinds)
            )
            metrics[f"guidance/scored_{kind}_count"] = sum(flag and value == kind for flag, value in zip(scored, kinds))
        return teacher, metrics

    def _finalize_training_teacher_log_probs(self, batch, sampled_teacher_log_probs):
        """Finalize candidate weights after the ordinary-input scoring pass."""
        response_shape = batch.batch["responses"].shape
        target_shape = (*response_shape, batch.meta_info["num_content_samples"])
        if sampled_teacher_log_probs.shape != response_shape:
            raise ValueError("Teacher probabilities must retain the original response positions")
        for key in ("content_target_ids", "content_teacher_log_probs", "content_old_log_probs"):
            if batch.batch[key].shape != target_shape:
                raise ValueError("Teacher labels and probabilities must retain every candidate position")
        score_mask = batch.batch["teacher_score_mask"].bool()
        selected_candidates = score_mask.unsqueeze(-1).expand(target_shape)
        teacher = batch.batch["old_log_probs"].detach().float().clone()
        active_rows = torch.as_tensor(batch.non_tensor_batch["teacher_score_active"], device=teacher.device)
        teacher[active_rows] = sampled_teacher_log_probs.to(teacher)[active_rows]
        if (not torch.isfinite(teacher[score_mask]).all()
                or not torch.isfinite(batch.batch["old_log_probs"][score_mask]).all()
                or not torch.isfinite(batch.batch["content_teacher_log_probs"][selected_candidates]).all()
                or not torch.isfinite(batch.batch["content_old_log_probs"][selected_candidates]).all()):
            raise FloatingPointError("Scored teacher and old log probabilities must be finite")
        original_gaps = teacher - batch.batch["old_log_probs"].float()
        return teacher, {
            "guidance/original_response_positive_gap_tokens": int((score_mask & (original_gaps > 0)).sum().item()),
            "guidance/original_response_negative_gap_tokens": int((score_mask & (original_gaps < 0)).sum().item()),
            "guidance/original_response_zero_gap_tokens": int((score_mask & (original_gaps == 0)).sum().item()),
            "guidance/original_response_gap_mean": float(original_gaps[score_mask].mean().item()) if score_mask.any() else 0.0,
        }

    def _build_obs_distill_inputs(self, batch):
        if batch.meta_info.get("content_weights_cached", False):
            raise RuntimeError("Teacher guidance must be applied only once per update")
        mask = batch.batch["teacher_score_mask"] * batch.batch["training_row_mask"][:, None]
        batch.batch["content_target_ids"] = batch.batch["content_target_ids"].detach().clone()
        batch.batch["content_teacher_log_probs"] = batch.batch["content_teacher_log_probs"].detach().float().clone()
        batch.batch["content_old_log_probs"] = batch.batch["content_old_log_probs"].detach().float().clone()
        weights = compute_content_weights(
            batch.batch["content_teacher_log_probs"], batch.batch["content_old_log_probs"],
            mask, batch.meta_info["guidance_beta"],
        )
        batch.batch["content_weights"] = weights.detach().clone()
        batch.meta_info["content_weights_cached"] = True
        selected = mask.bool()
        scored_rows = selected.any(dim=-1)
        zero_advantage = (batch.batch["advantages"] == 0).all(dim=-1)
        return batch, {
            "guidance/coef": batch.meta_info["guidance_coef"],
            "guidance/beta": batch.meta_info["guidance_beta"],
            "guidance/content_tokens": int(selected.sum().item()),
            "guidance/content_weight_mean": float(weights[selected].mean().item()) if selected.any() else 0.0,
            "guidance/content_samples_per_position": batch.meta_info["num_content_samples"],
            "guidance/zero_advantage_actions": int((scored_rows & zero_advantage).sum().item()),
        }

    def _apply_training_guidance(self, batch):
        return self._build_obs_distill_inputs(batch)
