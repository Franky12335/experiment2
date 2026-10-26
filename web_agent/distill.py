"""Sample and learn teacher next words under unchanged original prefixes."""

import math

import torch
import torch.nn.functional as F


def _validate_mask(mask, shape, device):
    if mask.shape != shape:
        raise ValueError("Mask must have the same shape as response positions")
    if mask.device != device:
        raise ValueError("Masks and scores must be on the same device")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must contain only zeros and ones")


def _checked_log_prob_values(values):
    if not values.is_floating_point():
        raise FloatingPointError("Log probabilities must be floating point values")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Selected log probabilities must be finite")
    if (values > 0).any():
        raise ValueError("Selected log probabilities must be nonpositive")
    values = values.float()
    if not torch.isfinite(values).all():
        raise FloatingPointError("Selected log probabilities must remain finite in FP32")
    return values


def _validate_logits(logits, mask):
    # Standard batches use [B, R, V]; packed actor forwards use [N, V].
    if not logits.is_floating_point() or logits.ndim < 2 or logits.shape[-1] == 0:
        raise ValueError("Logits must be floating point values with a nonempty vocabulary dimension")
    _validate_mask(mask, logits.shape[:-1], logits.device)
    return mask.bool()


def _candidate_mask(values, content_mask):
    """Broadcast response positions across the required candidate dimension."""
    if values.ndim != content_mask.ndim + 1 or values.shape[:-1] != content_mask.shape:
        raise ValueError("Candidate values must match positions and include a candidate dimension")
    if values.shape[-1] < 1:
        raise ValueError("Each position must have at least one candidate")
    _validate_mask(content_mask, values.shape[:-1], values.device)
    return content_mask.bool().unsqueeze(-1).expand_as(values)


def _normalized_logits(logits):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Selected logits must be finite")
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Selected logits must remain finite in FP32")
    normalized = F.log_softmax(logits, dim=-1)
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("Selected normalized logits must remain finite in FP32")
    return normalized


def compute_content_weights(teacher_log_probs, old_log_probs, content_mask, gate_beta):
    """Return detached FP32 sigmoid weights on the teacher's sampled words.

    Equal scores give weight one half. A negative difference reduces imitation
    strength without assigning a negative weight. Unselected scores are ignored.
    """
    if isinstance(gate_beta, bool) or not math.isfinite(gate_beta) or gate_beta <= 0:
        raise ValueError("gate_beta must be finite and positive")
    if teacher_log_probs.shape != old_log_probs.shape or teacher_log_probs.device != old_log_probs.device:
        raise ValueError("Teacher and old log probabilities must have matching shapes and devices")
    if not teacher_log_probs.is_floating_point() or not old_log_probs.is_floating_point():
        raise FloatingPointError("Teacher and old log probabilities must be floating point values")
    selected = _candidate_mask(old_log_probs, content_mask)
    teacher = _checked_log_prob_values(teacher_log_probs.detach()[selected])
    old = _checked_log_prob_values(old_log_probs.detach()[selected])
    weights = torch.zeros_like(old_log_probs, dtype=torch.float32)
    # FP64 multiplication keeps a large finite beta from becoming an infinite
    # scalar before multiplication by a zero score difference.
    weights[selected] = torch.sigmoid((teacher.double() - old.double()) * gate_beta).float()
    return weights.detach()


def sample_content_targets(logits, content_mask, num_samples=4):
    """Sample next words at each selected original response prefix.

    The logits and original prefix inputs are not changed. Masked positions
    return zero labels and scores. Empty masks do not consume random draws.
    Outputs append a candidate dimension to the mask shape. IDs are long and
    log scores are detached FP32 values. Candidates are drawn with replacement.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    selected = _validate_logits(logits, content_mask)
    output_shape = (*content_mask.shape, num_samples)
    target_ids = torch.zeros(output_shape, dtype=torch.long, device=logits.device)
    target_log_probs = torch.zeros(output_shape, dtype=torch.float32, device=logits.device)
    if not selected.any():
        return target_ids, target_log_probs
    positions = selected.nonzero(as_tuple=True)
    for start in range(0, positions[0].numel(), 64):
        indices = tuple(index[start:start + 64] for index in positions)
        normalized = _normalized_logits(logits.detach()[indices])
        sampled_ids = torch.multinomial(normalized.exp(), num_samples=num_samples, replacement=True)
        sampled_log_probs = normalized.gather(-1, sampled_ids)
        target_ids[indices] = sampled_ids
        target_log_probs[indices] = _checked_log_prob_values(sampled_log_probs)
    return target_ids, target_log_probs.detach()


def gather_content_log_probs(logits, target_ids, content_mask):
    """Score extra labels from logits computed under the original prefixes.

    No label is substituted into the model's prefix inputs. Current logits keep
    their gradient; masked logits and labels are neither inspected nor trained.
    """
    selected = _validate_logits(logits, content_mask)
    _candidate_mask(target_ids, content_mask)
    if target_ids.dtype != torch.long:
        raise ValueError("Target IDs must have integer long dtype")
    # An empty slice keeps a zero-gradient connection without reading masked NaNs.
    result = logits.reshape(-1, logits.shape[-1])[:0].float().sum().expand(target_ids.shape).clone()
    if not selected.any():
        return result
    selected_ids = target_ids[selected]
    if ((selected_ids < 0) | (selected_ids >= logits.shape[-1])).any():
        raise ValueError("Selected target IDs must be inside the vocabulary")
    positions = selected.nonzero(as_tuple=True)
    for start in range(0, positions[0].numel(), 64):
        indices = tuple(index[start:start + 64] for index in positions)
        normalized = _normalized_logits(logits[indices])
        labels = target_ids[indices]
        scores = normalized.gather(-1, labels)
        result[indices] = _checked_log_prob_values(scores)
    return result


def compute_content_distill_loss(content_log_probs, content_weights, content_mask, response_mask,
                                 normalization="content_positions"):
    """Average candidates and use the denominator selected by the model profile.

    The caller excludes artificial rows in response_mask. Non-content response
    positions enter the response denominator and are excluded from the content
    denominator. Weights are detached and bounded in
    [0, 1]; zero weights do not inspect their scores. Multiplying this finite loss
    by a zero coefficient gives zero gradient without changing task advantages.
    """
    if not content_log_probs.is_floating_point() or not content_weights.is_floating_point():
        raise FloatingPointError("Log probabilities and weights must be floating point values")
    if content_weights.shape != content_log_probs.shape or content_weights.device != content_log_probs.device:
        raise ValueError("Weights must match the log probability shape and device")
    selected = _candidate_mask(content_log_probs, content_mask)
    _validate_mask(response_mask, content_mask.shape, content_log_probs.device)
    if (content_mask.bool() & ~response_mask.bool()).any():
        raise ValueError("Content positions must be valid response positions")
    selected_weights = content_weights.detach()[selected]
    if not torch.isfinite(selected_weights).all():
        raise FloatingPointError("Selected weights must be finite")
    if ((selected_weights < 0) | (selected_weights > 1)).any():
        raise ValueError("Selected weights must lie between zero and one")
    fixed_weights = torch.zeros_like(content_log_probs, dtype=torch.float32)
    fixed_weights[selected] = selected_weights.float()
    active = selected & (fixed_weights > 0)
    scores = _checked_log_prob_values(content_log_probs[active])
    terms = content_log_probs.reshape(-1)[:0].float().sum().expand(content_log_probs.shape).clone()
    terms[active] = -fixed_weights[active] * scores
    if normalization not in ("content_positions", "response_positions"):
        raise ValueError("Unknown content objective normalization")
    num_candidates = content_log_probs.shape[-1]
    denominator = content_mask if normalization == "content_positions" else response_mask
    loss = (terms.double().sum() / num_candidates / denominator.sum().clamp_min(1)).float()
    if not torch.isfinite(loss):
        raise FloatingPointError("Content loss must remain finite in FP32")
    return loss


