# Modified for the anonymous main training and evaluation release.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import math
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from web_agent.distill import compute_content_distill_loss, gather_content_log_probs
from web_agent.distill import sample_content_targets as draw_content_targets

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.profiling import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False,
                             sample_content_targets=False, compute_content_log_probs=False,
                             num_content_samples=4):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        content_requested = sample_content_targets or compute_content_log_probs
        if sample_content_targets and compute_content_log_probs:
            raise ValueError("Content sampling and label scoring are mutually exclusive")
        if content_requested and self.use_fused_kernels:
            raise ValueError("Content labels require model logits; disable use_fused_kernels")
        content_extras = {}
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if content_requested:
                content_mask = micro_batch["teacher_score_mask"]
                if content_mask.shape != micro_batch["responses"].shape or content_mask.device != input_ids.device:
                    raise ValueError("Content mask must match the original response positions")
                if not torch.all((content_mask == 0) | (content_mask == 1)):
                    raise ValueError("Content mask must contain only zeros and ones")
                if (content_mask.bool() & ~attention_mask[:, -response_length:].bool()).any():
                    raise ValueError("Content positions must be valid response positions")
                if compute_content_log_probs:
                    target_ids = micro_batch["content_target_ids"]
                    if (target_ids.ndim != content_mask.ndim + 1
                            or target_ids.shape[:-1] != content_mask.shape
                            or target_ids.shape[-1] < 1
                            or target_ids.device != input_ids.device):
                        raise ValueError("Content labels must match response positions and candidate count")
                    if target_ids.dtype != torch.long:
                        raise ValueError("Content labels must have integer long dtype")
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                if content_requested:
                    full_content_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                    full_content_mask[:, -response_length:] = content_mask.bool()
                    content_mask_rmpad = index_first_axis(full_content_mask.reshape(-1, 1), indices).transpose(0, 1)
                    content_mask_rmpad = torch.roll(content_mask_rmpad, shifts=-1, dims=1)
                    if compute_content_log_probs:
                        num_target_candidates = target_ids.shape[-1]
                        full_target_ids = input_ids.new_zeros((batch_size, seqlen, num_target_candidates))
                        full_target_ids[:, -response_length:] = target_ids
                        content_ids_rmpad = index_first_axis(
                            full_target_ids.reshape(-1, num_target_candidates), indices).unsqueeze(0)
                        content_ids_rmpad = torch.roll(content_ids_rmpad, shifts=-1, dims=1)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    if content_requested:
                        content_mask_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                            content_mask_rmpad, position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        if compute_content_log_probs:
                            # The sequence helper slices dimension 1 of a matrix.
                            # Treat candidates as rows while slicing positions.
                            content_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                                content_ids_rmpad.squeeze(0).transpose(0, 1), position_ids_rmpad=None,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )
                            content_ids_rmpad = content_ids_rmpad.transpose(0, 1).unsqueeze(0)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or content_requested:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    if sample_content_targets:
                        sampled_ids, sampled_log_probs = draw_content_targets(
                            logits_rmpad.unsqueeze(0), content_mask_rmpad, num_samples=num_content_samples)
                        content_extras = {
                            "content_target_ids": sampled_ids.squeeze(0),
                            "content_teacher_log_probs": sampled_log_probs.squeeze(0),
                        }
                    elif compute_content_log_probs:
                        content_extras["content_log_probs"] = gather_content_log_probs(
                            logits_rmpad.unsqueeze(0), content_ids_rmpad, content_mask_rmpad).squeeze(0)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    content_extras = {
                        key: gather_outpus_and_unpad(value, gather_dim=0, unpad_dim=0,
                                                    padding_size=pad_size)
                        for key, value in content_extras.items()
                    }
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                content_extras = {
                    key: pad_input(value, indices, batch_size, seqlen)[:, -response_length - 1 : -1]
                    for key, value in content_extras.items()
                }

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"],
                                                    inplace_backward=not (calculate_entropy or content_requested))
                    if sample_content_targets:
                        sampled_ids, sampled_log_probs = draw_content_targets(
                            logits, content_mask, num_samples=num_content_samples)
                        content_extras = {"content_target_ids": sampled_ids,
                                          "content_teacher_log_probs": sampled_log_probs}
                    elif compute_content_log_probs:
                        content_extras["content_log_probs"] = gather_content_log_probs(logits, target_ids, content_mask)
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            if content_requested:
                return entropy, log_probs, content_extras
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        sample_targets = data.meta_info.get("sample_content_targets", False)
        score_targets = data.meta_info.get("compute_content_log_probs", False)
        if sample_targets and score_targets:
            raise ValueError("Content sampling and label scoring are mutually exclusive")
        content_requested = sample_targets or score_targets
        content_args = {}
        if sample_targets:
            content_args["sample_content_targets"] = True
            content_args["num_content_samples"] = data.meta_info.get("num_content_samples", 4)
        elif score_targets:
            content_args["compute_content_log_probs"] = True
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if content_requested:
            select_keys.append("teacher_score_mask")
        if score_targets:
            select_keys.append("content_target_ids")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        content_lists = {}
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                forward_output = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, **content_args)
                entropy, log_probs = forward_output[:2]
                if content_requested:
                    for key, value in forward_output[2].items():
                        content_lists.setdefault(key, []).append(value.detach())
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        content_extras = {key: torch.concat(values, dim=0) for key, values in content_lists.items()}
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]
            content_extras = {key: value[revert_indices] for key, value in content_extras.items()}
        if content_requested:
            return log_probs, entropys, content_extras
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        content_keys = ("content_target_ids", "content_weights", "teacher_score_mask")
        content_present = [key in data.batch for key in content_keys]
        if any(content_present) and not all(content_present):
            raise ValueError("Content updates require target labels, weights and the content mask")
        use_content_distill = all(content_present)
        if use_content_distill:
            num_content_samples = data.meta_info["num_content_samples"]
            if isinstance(num_content_samples, bool) or not isinstance(num_content_samples, int) or num_content_samples < 1:
                raise ValueError("num_content_samples must be a positive integer")
            target_shape = data.batch["content_target_ids"].shape
            response_shape = data.batch["responses"].shape
            if (len(target_shape) != len(response_shape) + 1
                    or target_shape[:-1] != response_shape
                    or target_shape[-1] != num_content_samples):
                raise ValueError("Content labels must retain the configured candidate count")
        content_coef = float(data.meta_info["guidance_coef"]) if use_content_distill else 0.0
        content_normalization = data.meta_info["normalization"] if use_content_distill else "content_positions"
        if use_content_distill and (not math.isfinite(content_coef) or content_coef < 0):
            raise ValueError("Content coefficient must be finite and nonnegative")
        has_training_row_mask = "training_row_mask" in data.batch
        use_update_normalization = has_training_row_mask or use_content_distill
        if use_update_normalization:
            from web_agent.gradient_update import UpdateNormalization
            if self.use_ulysses_sp:
                raise NotImplementedError("Update normalization requires data parallelism without sequence parallelism")
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if use_content_distill:
            select_keys.extend(content_keys)
        if has_training_row_mask:
            select_keys.append("training_row_mask")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            if use_update_normalization:
                batch = batch[torch.randperm(batch.batch_size[0])]
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                if use_update_normalization:
                    mini_tensors = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    update_response_length = mini_tensors["responses"].size(-1)
                    update_mask_key = "loss_mask" if multi_turn else "attention_mask"
                    update_mask = mini_tensors[update_mask_key][:, -update_response_length:]
                    if has_training_row_mask:
                        update_mask = update_mask * mini_tensors["training_row_mask"][:, None]
                    content_tokens = ((mini_tensors["teacher_score_mask"] * update_mask).sum().item()
                                      if use_content_distill else 0)
                    normalization = UpdateNormalization(mini_tensors["input_ids"].device,
                                                        update_mask.bool().any(dim=-1).sum().item(),
                                                        update_mask.sum().item(), content_tokens=content_tokens,
                                                        normalization=content_normalization)
                    if not normalization.active:
                        continue
                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    if has_training_row_mask:
                        response_mask = response_mask * data["training_row_mask"][:, None]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    forward_args = {}
                    forward_batch = data
                    if use_content_distill:
                        forward_args["compute_content_log_probs"] = True
                        forward_batch = {**data, "teacher_score_mask": data["teacher_score_mask"] * response_mask}
                    forward_output = self._forward_micro_batch(
                        micro_batch=forward_batch, temperature=temperature,
                        calculate_entropy=calculate_entropy, **forward_args)
                    entropy, log_prob = forward_output[:2]
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef


                    if use_content_distill:
                        content_loss = compute_content_distill_loss(
                            forward_output[2]["content_log_probs"], data["content_weights"],
                            forward_batch["teacher_score_mask"], response_mask,
                            normalization=content_normalization)
                        weighted_content_loss = content_coef * content_loss
                        append_to_dict(metrics, {
                            "actor/content_distill_loss": content_loss.detach().item(),
                            "actor/content_distill_weighted_loss": weighted_content_loss.detach().item(),
                            "actor/content_distill_coef": content_coef,
                        })
                    if use_update_normalization:
                        policy_scale, _ = normalization.scales(response_mask, loss_agg_mode)
                        loss = policy_loss * policy_scale
                        if use_content_distill:
                            content_scale = normalization.content_scale(forward_batch["teacher_score_mask"], response_mask)
                            loss = loss + weighted_content_loss * content_scale
                    else:
                        if self.config.use_dynamic_bsz:
                            policy_scale = len(data) / self.config.ppo_mini_batch_size
                        else:
                            policy_scale = 1.0 / self.gradient_accumulation
                        loss = policy_loss * policy_scale

                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                if use_update_normalization:
                    del normalization
                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

