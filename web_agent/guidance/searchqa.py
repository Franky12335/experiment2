from numbers import Number

from web_agent.guidance.records import fact


def _items(value):
    if value is None:
        return []
    if isinstance(value, (str, Number)):
        return [value]
    return list(value)


def _answer_aliases(env_kwargs):
    truth = (env_kwargs or {}).get("ground_truth") or {}
    target = truth.get("target") if isinstance(truth, dict) else truth
    return [str(item).strip() for item in _items(target) if str(item).strip()]


def search_facts(candidates, metadata, env_kwargs, turn, visible_topk):
    """Keep complete actual documents shown to the ordinary policy."""
    output = []
    for item in candidates:
        rank = int(item["rank"])
        raw = item["text"]
        content = raw.strip() if isinstance(raw, str) else raw
        visible = 1 <= rank <= visible_topk and item.get("student_visible", True) is True
        if not visible or not isinstance(content, str) or not content.strip():
            continue
        source_id = str(item["document_id"])
        start, end = len(raw) - len(raw.lstrip()), len(raw.rstrip())
        record = fact(content, "search_document",
                      f"turn={turn},rank={rank},id={source_id},chars={start}:{end}",
                      turn, True, "retrieved_material", source_id=source_id, source_verified=True)
        record["rank"] = rank
        output.append(record)
    return output


def search_reference(metadata, env_kwargs):
    return {"answer_aliases": _answer_aliases(env_kwargs)}
