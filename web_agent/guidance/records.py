import json
import re
from dataclasses import dataclass


@dataclass
class MaterialSelection:
    reason: str
    facts: list

    @property
    def available(self):
        return bool(self.facts)


def unpack(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def pack(value):
    return json.dumps(value, ensure_ascii=False)


def material_in_observation(text, observation, task):
    """Find a complete actual material after lossless presentation unwrapping."""
    if not isinstance(text, str) or not text.strip() or not isinstance(observation, str):
        return False
    if text in observation:
        return True
    if task == "searchqa":
        decoder = json.JSONDecoder()
        for match in re.finditer(r"<information>\s*", observation):
            try:
                payload, end = decoder.raw_decode(observation, match.end())
            except (TypeError, ValueError):
                continue
            if not observation[end:].lstrip().startswith("</information>"):
                continue
            result = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(result, str) and text in result:
                return True
    return False


def fact(text, source, position, turn, visible, relation, *, source_id, value=None,
         source_verified=False):
    record = {
        "text": str(text),
        "source": source,
        "position": str(position),
        "first_turn": int(turn),
        "observed_turn": int(turn),
        "first_visible_turn": int(turn) if visible else None,
        "is_repeat": False,
        "student_visible": bool(visible),
        "relation": relation,
        "source_id": str(source_id),
        "source_verified": bool(source_verified),
    }
    if value is not None:
        record["value"] = str(value)
    return record


def fact_key(item):
    key = (item["source"], item["source_id"] or item["position"],
           item["relation"], item["value"] if "value" in item else item["text"])
    return key


def annotate_occurrences(known, returned):
    previous = {fact_key(item): item for item in known}
    output = []
    for item in returned:
        record = dict(item)
        earlier = previous.get(fact_key(item))
        if earlier is not None:
            record["first_turn"] = earlier["first_turn"]
            record["first_visible_turn"] = earlier["first_visible_turn"]
            if record["first_visible_turn"] is None and record["student_visible"]:
                record["first_visible_turn"] = record["observed_turn"]
            record["is_repeat"] = True
        output.append(record)
        previous[fact_key(record)] = record
    return output


def merge_records(known, returned):
    output = list(known)
    positions = {fact_key(item): index for index, item in enumerate(output)}
    for item in returned:
        key = fact_key(item)
        if key in positions:
            output[positions[key]] = item
        else:
            positions[key] = len(output)
            output.append(item)
    return output


def append_new(seen, returned):
    return merge_records(seen, [item for item in returned if item["student_visible"]])
