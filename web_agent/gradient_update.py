"""Preserve objective averages when splitting an optimizer update into microbatches."""

import torch
import torch.distributed as dist


class UpdateNormalization:

    def __init__(self, device, rows, response_tokens, content_tokens=0,
                 normalization="content_positions"):
        if normalization not in ("content_positions", "response_positions"):
            raise ValueError("Unknown content objective normalization")
        self.normalization = normalization
        self.denominators = torch.tensor([rows, response_tokens, content_tokens], dtype=torch.float64, device=device)
        if not torch.isfinite(self.denominators).all() or (self.denominators < 0).any():
            raise ValueError("Optimizer update counts must be finite and nonnegative")
        if content_tokens > response_tokens:
            raise ValueError("Content positions cannot exceed response positions")
        self.world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.denominators, op=dist.ReduceOp.SUM)
            self.world_size = dist.get_world_size()
        if not torch.isfinite(self.denominators).all() or (self.denominators < 0).any():
            raise ValueError("Optimizer update counts must be finite and nonnegative")
        if (self.denominators[:2] == 0).any() and (self.denominators[:2] > 0).any():
            raise ValueError("Training rows must contain valid response tokens")
        if self.denominators[2] > self.denominators[1]:
            raise ValueError("Content positions cannot exceed response positions")
        self.active = bool(self.denominators[1] > 0)

    def scales(self, response_mask, loss_agg_mode):
        if not self.active:
            return 0.0, 0.0
        active_rows = response_mask.bool().any(dim=-1).sum()
        action_scale = active_rows * self.world_size / self.denominators[0]
        if loss_agg_mode == "token-mean":
            policy_scale = response_mask.sum() * self.world_size / self.denominators[1]
        elif loss_agg_mode in ("seq-mean-token-mean", "seq-mean-token-sum"):
            policy_scale = action_scale
        elif loss_agg_mode == "seq-mean-token-sum-norm":
            policy_scale = self.denominators.new_tensor(self.world_size)
        else:
            raise ValueError(f"Unsupported policy aggregation: {loss_agg_mode}")
        return float(policy_scale.item()), float(action_scale.item())

    def content_scale(self, content_mask, response_mask=None):
        if not torch.all((content_mask == 0) | (content_mask == 1)):
            raise ValueError("Content masks must contain only zeros and ones")
        content_tokens = content_mask.sum()
        if content_tokens > self.denominators[2]:
            raise ValueError("Microbatch content positions exceed the update total")
        if self.normalization == "response_positions":
            if response_mask is None or response_mask.shape != content_mask.shape:
                raise ValueError("Response normalization requires the matching response mask")
            if not torch.all((response_mask == 0) | (response_mask == 1)):
                raise ValueError("Response masks must contain only zeros and ones")
            if (content_mask.bool() & ~response_mask.bool()).any():
                raise ValueError("Content positions must be valid response positions")
            response_tokens = response_mask.sum()
            if response_tokens > self.denominators[1]:
                raise ValueError("Microbatch response positions exceed the update total")
            if self.denominators[1] == 0:
                return 0.0
            return float((response_tokens * self.world_size / self.denominators[1]).item())
        if self.denominators[2] == 0:
            return 0.0
        return float((content_tokens * self.world_size / self.denominators[2]).item())
