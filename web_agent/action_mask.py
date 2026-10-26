"""Locate the environment action in an assistant response."""

import codecs
import re
from dataclasses import dataclass

import torch
from tokenizers.decoders import ByteLevel


SEARCH_QUERY = re.compile(r"<search>.*?</search>", re.I | re.S)
SEARCH_ANSWER = re.compile(r"<answer>.*?</answer>", re.I | re.S)


def _byte_values() -> dict[str, int]:
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(map(chr, characters), values))


BYTE_VALUES = _byte_values()


@dataclass(frozen=True)
class ParsedAction:
    response: str
    span: tuple[int, int]
    body_mask: torch.Tensor
    body_text: str


class ActionDecoder:
    """Reuse vocabulary bytes to locate command content in original response IDs."""

    def __init__(self, tokenizer):
        if not isinstance(tokenizer.backend_tokenizer.decoder, ByteLevel):
            raise ValueError("Action decoding requires a ByteLevel tokenizer decoder")
        self.tokenizer = tokenizer
        self.token_bytes = {}
        for token, token_id in tokenizer.get_vocab().items():
            # ByteLevel interprets a complete token as UTF-8 when it contains
            # a character outside its byte alphabet, including added tokens.
            if all(character in BYTE_VALUES for character in token):
                data = bytes(BYTE_VALUES[character] for character in token)
            else:
                data = token.encode("utf-8")
            self.token_bytes[token_id] = data

    def parse(self, response_ids: torch.Tensor, response_mask: torch.Tensor, task: str) -> ParsedAction:
        if task != "searchqa":
            raise ValueError(f"Unknown task: {task}")
        if response_ids.ndim != 1 or response_ids.shape != response_mask.shape:
            raise ValueError("Response IDs and response mask must be equal one-dimensional tensors")
        positions = torch.nonzero(response_mask.bool(), as_tuple=False).flatten()
        valid_ids = response_ids[positions].tolist()
        response = self.tokenizer.decode(valid_ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        span = action_span(response, task)
        mask = torch.zeros_like(response_mask)
        start, end = span
        if start == end:
            return ParsedAction(response, span, mask, "")
        tag = _action_tag(response, span, task)
        raw_start = start + len(tag) + 2
        raw_end = end - len(tag) - 3
        body_text = response[raw_start:raw_end]
        if not body_text.strip():
            return ParsedAction(response, span, mask, body_text)

        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        completed_length = 0
        previous_length = 0
        previous_boundary = 0
        selected = [0] * len(valid_ids)
        for boundary, token_id in enumerate(valid_ids, start=1):
            # The tokenizer omits IDs without a vocabulary entry. They retain
            # their original response positions without adding decoded bytes.
            if token_id in self.token_bytes:
                completed_length += len(decoder.decode(self.token_bytes[token_id], final=False))
            pending = decoder.getstate()[0]
            pending_text = pending.decode("utf-8", errors="replace")
            prefix_length = completed_length + len(pending_text)
            # At most three bytes await completion. Decode that small suffix
            # as an ended prefix, including invalid Unicode sequences, and
            # retain its boundary when the final text has the same suffix.
            stable = response[completed_length:prefix_length] == pending_text
            if not stable or prefix_length <= previous_length:
                continue
            intersection_start = max(previous_length, raw_start)
            intersection_end = min(prefix_length, raw_end)
            if (intersection_start < intersection_end
                    and response[intersection_start:intersection_end].strip()):
                selected[previous_boundary:boundary] = [1] * (boundary - previous_boundary)
            previous_length = prefix_length
            previous_boundary = boundary
        mask[positions] = torch.tensor(selected, dtype=mask.dtype, device=mask.device)
        return ParsedAction(response, span, mask, body_text)


def _action_tag(response: str, span: tuple[int, int], task: str) -> str:
    if task != "searchqa":
        raise ValueError(f"Unknown task: {task}")
    return "search" if response[span[0]:span[0] + len("<search>")].lower() == "<search>" else "answer"


def action_span(response: str, task: str) -> tuple[int, int]:
    if task != "searchqa":
        raise ValueError(f"Unknown task: {task}")
    match = SEARCH_QUERY.search(response)
    if match is None:
        match = SEARCH_ANSWER.search(response)
    return match.span() if match is not None else (0, 0)


def matches_projected_action(response: str, projected_action: str, task: str, *, span: tuple[int, int]) -> bool:
    if task != "searchqa":
        raise ValueError(f"Unknown task: {task}")
    start, end = span
    if start == end:
        return False
    action = response[start:end]
    tag = _action_tag(response, span, task)
    expected = f"<{tag}>{action[len(tag) + 2:-(len(tag) + 3)].strip()}</{tag}>"
    return expected == str(projected_action)


