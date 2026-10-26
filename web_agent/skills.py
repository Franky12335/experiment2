"""Retrieve candidate task principles for an unchanged original action."""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re


TASK_ACTIONS = {'searchqa': ('search', 'answer')}
_STATE_FIELDS = {"task", "task_text", "observation_text", "turn", "remaining_actions"}
_CARD_FIELDS = {"skill_id", "title", "principle", "when_to_apply"}
_SOURCE_FILES = {'searchqa': 'search/claude_style_skills_search.json'}
_CATEGORIES = {'searchqa': ('direct_retrieval',
              'multi_hop_reasoning',
              'entity_attribute_lookup',
              'comparison')}
# These identifiers limit action candidates without altering source text.
_ACTIONS = {'searchqa': {'gen_001': ('search',),
              'gen_002': ('search',),
              'gen_003': ('search',),
              'gen_004': ('answer',),
              'gen_005': ('search', 'answer'),
              'gen_006': ('search',),
              'gen_007': ('search',),
              'gen_008': ('search', 'answer'),
              'gen_009': ('search',),
              'gen_010': ('answer',),
              'dir_001': ('search',),
              'dir_002': ('search',),
              'dir_003': ('search',),
              'dir_004': ('search',),
              'dir_005': ('answer',),
              'mul_001': ('search',),
              'mul_002': ('search',),
              'mul_003': ('search', 'answer'),
              'mul_004': ('search',),
              'mul_005': ('search',),
              'ent_001': ('search',),
              'ent_002': ('search',),
              'ent_003': ('search', 'answer'),
              'ent_004': ('search',),
              'ent_005': ('answer',),
              'com_001': ('search',),
              'com_002': ('search',),
              'com_003': ('answer',),
              'com_004': ('search',),
              'com_005': ('answer',)}}
_PREFIXES = {
    "general_skills": "gen",
    "direct_retrieval": "dir", "multi_hop_reasoning": "mul",
    "entity_attribute_lookup": "ent", "comparison": "com",
}
_STOP_WORDS = frozenset(
    "a an and are as at be before between by can do does for from has have if in into is it "
    "its of on once only or should than that the their them then these this those to upon "
    "use using was when whenever where which while who with without you your".split()
)


def _exact_fields(value, expected, name):
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        interval = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{name} must be an integer in {interval}")


def _text(value, name, allow_empty=False):
    if type(value) is not str or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be {'a' if allow_empty else 'a nonempty'} string")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated JSON field: {key}")
        result[key] = value
    return result


def _read_json(path):
    if not path.is_file():
        raise ValueError(f"Missing skill resource: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_json_object)


def _tokens(text):
    words = []
    for word in re.findall(r"[a-z0-9]+", text.casefold()):
        if word in _STOP_WORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
            word = word[:-1]
        words.append(word)
    return words


def _validate_mapping(document, task):
    if task != "searchqa":
        raise ValueError(f"Unknown skill task: {task}")
    _exact_fields(document, {"skill_files", "task_to_skill"}, "SearchQA skill mapping")
    mapped_categories = tuple("compare" if category == "comparison" else category for category in _CATEGORIES[task])
    expected_files = {category: f"{category}.md" for category in ("general_skills", *mapped_categories)}
    if document["skill_files"] != expected_files:
        raise ValueError("SearchQA skill_files must identify the existing category documents")
    if document["task_to_skill"] != {category: category for category in mapped_categories}:
        raise ValueError("SearchQA task_to_skill must identify every existing category")


def _load_task_skills(document, task):
    category_field = "query_type_skills"
    _exact_fields(
        document, {"general_skills", category_field, "common_mistakes", "metadata"},
        f"{task} skill document",
    )
    if type(document["common_mistakes"]) is not list or type(document["metadata"]) is not dict:
        raise ValueError(f"{task} common_mistakes and metadata must retain their source structure")
    grouped = document[category_field]
    _exact_fields(grouped, set(_CATEGORIES[task]), f"{task} categories")
    skills = []
    for category, records in (("general_skills", document["general_skills"]), *grouped.items()):
        if type(records) is not list or not records:
            raise ValueError(f"{task}/{category} must contain existing skill records")
        for item in records:
            _exact_fields(item, _CARD_FIELDS, f"{task}/{category} skill")
            for field in _CARD_FIELDS:
                _text(item[field], f"{task}/{category}.{field}")
            identifier = item["skill_id"]
            if identifier not in _ACTIONS[task] or not identifier.startswith(_PREFIXES[category] + "_"):
                raise ValueError(f"Unknown identifier or category: {task}/{category}/{identifier}")
            skills.append(Skill(
                skill_id=identifier,
                task=task,
                actions=_ACTIONS[task][identifier],
                instruction=item["principle"],
                title=item["title"],
                when_to_apply=item["when_to_apply"],
                category=category,
                source_file=_SOURCE_FILES[task],
            ))
    identifiers = [skill.skill_id for skill in skills]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{task} contains repeated skill identifiers")
    if set(identifiers) != set(_ACTIONS[task]):
        raise ValueError(f"{task} is missing existing skill identifiers")
    return tuple(skills)


@dataclass(frozen=True, slots=True)
class Skill:
    """One existing principle, its application text, and its exact source identity."""

    skill_id: str
    task: str
    actions: tuple[str, ...]
    instruction: str
    title: str
    when_to_apply: str
    category: str
    source_file: str

    def __post_init__(self):
        for name in ("skill_id", "task", "instruction", "title", "when_to_apply", "category", "source_file"):
            _text(getattr(self, name), name)
        if self.task not in TASK_ACTIONS:
            raise ValueError(f"Unknown skill task: {self.task}")
        if re.fullmatch(r"[a-z]{3}_[0-9]{3}", self.skill_id) is None:
            raise ValueError("skill_id must retain its existing identifier")
        if self.category not in ("general_skills", *_CATEGORIES[self.task]):
            raise ValueError(f"Unknown category {self.category} for task {self.task}")
        if self.source_file != _SOURCE_FILES[self.task]:
            raise ValueError(f"Unexpected principle source: {self.source_file}")
        if type(self.actions) is not tuple:
            raise ValueError("actions must be an immutable sequence")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions contains repeated entries")
        for action in self.actions:
            if type(action) is not str or action not in TASK_ACTIONS[self.task]:
                raise ValueError(f"Unknown action {action} for task {self.task}")

    def record(self):
        return {
            "skill_id": self.skill_id,
            "task": self.task,
            "actions": list(self.actions),
            "instruction": self.instruction,
            "title": self.title,
            "when_to_apply": self.when_to_apply,
            "category": self.category,
            "source_file": self.source_file,
        }


@dataclass(frozen=True, slots=True)
class _SkillStatistics:
    skill: Skill
    terms: tuple[str, ...]
    counts: tuple[int, ...]
    length: int
    application_terms: frozenset[str]


@dataclass(frozen=True, slots=True)
class _CandidateStatistics:
    documents: tuple[_SkillStatistics, ...]
    terms: tuple[str, ...]
    document_frequencies: tuple[int, ...]
    inverse_frequencies: tuple[float, ...]
    average_length: float
    length_adjustments: tuple[float, ...]


def _skill_statistics(skill):
    document = (
        Counter(_tokens(skill.instruction))
        + Counter({term: count * 2 for term, count in Counter(_tokens(skill.title)).items()})
        + Counter({term: count * 2 for term, count in Counter(_tokens(skill.when_to_apply)).items()})
    )
    terms = tuple(sorted(document))
    return _SkillStatistics(
        skill=skill,
        terms=terms,
        counts=tuple(document[term] for term in terms),
        length=sum(document.values()),
        application_terms=frozenset(_tokens(skill.when_to_apply)),
    )


def _candidate_statistics(statistics, task, action):
    documents = tuple(
        entry for entry in statistics
        if entry.skill.task == task and action in entry.skill.actions
    )
    frequencies = Counter(term for document in documents for term in document.terms)
    terms = tuple(sorted(frequencies))
    average_length = sum(document.length for document in documents) / len(documents) if documents else 0.0
    return _CandidateStatistics(
        documents=documents,
        terms=terms,
        document_frequencies=tuple(frequencies[term] for term in terms),
        inverse_frequencies=tuple(
            math.log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            for term in terms
        ),
        average_length=average_length,
        length_adjustments=tuple(
            1.2 * (0.25 + 0.75 * document.length / average_length) for document in documents
        ),
    )


@dataclass(frozen=True, slots=True)
class SkillBank:
    skills: tuple[Skill, ...]
    max_skills: int
    resource_paths: tuple[Path, ...]
    _candidates: tuple[_CandidateStatistics, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _integer(self.max_skills, "max_skills", 0, 3)
        if type(self.skills) is not tuple or any(type(skill) is not Skill for skill in self.skills):
            raise ValueError("SkillBank must contain immutable existing Skills")
        identities = [(skill.task, skill.skill_id) for skill in self.skills]
        if len(identities) != len(set(identities)):
            raise ValueError("Skill identities must be unique within their task")
        for task in TASK_ACTIONS:
            if {skill.skill_id for skill in self.skills if skill.task == task} != set(_ACTIONS[task]):
                raise ValueError(f"SkillBank must retain every existing {task} principle")
        if type(self.resource_paths) is not tuple or len(self.resource_paths) != 2:
            raise ValueError("SkillBank must identify its two source JSON resources")
        if any(not isinstance(path, Path) for path in self.resource_paths):
            raise ValueError("Skill resource paths must be Paths")
        statistics = tuple(_skill_statistics(skill) for skill in self.skills)
        object.__setattr__(self, "_candidates", tuple(
            _candidate_statistics(statistics, "searchqa", action) for action in TASK_ACTIONS["searchqa"]
        ))

    @classmethod
    def load(cls, directory, max_skills):
        _integer(max_skills, "max_skills", 0, 3)
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Skill library must be a directory: {root}")
        paths = (root / _SOURCE_FILES["searchqa"], root / "search/skill_mapping.json")
        document, mapping = (_read_json(path) for path in paths)
        _validate_mapping(mapping, "searchqa")
        return cls(skills=_load_task_skills(document, "searchqa"), max_skills=max_skills, resource_paths=paths)

    @staticmethod
    def validate_state(state):
        _exact_fields(state, _STATE_FIELDS, "Action state")
        _text(state["task"], "state.task")
        if state["task"] not in TASK_ACTIONS:
            raise ValueError(f"Unknown state task: {state['task']}")
        _text(state["task_text"], "state.task_text")
        _text(state["observation_text"], "state.observation_text", allow_empty=True)
        _integer(state["turn"], "state.turn", 0)
        _integer(state["remaining_actions"], "state.remaining_actions", 1)


    def _rank(self, candidates, state, original_action):
        query = Counter({term: count * 2 for term, count in Counter(_tokens(state["task_text"])).items()})
        query.update(_tokens(original_action))
        query_terms = []
        for term, query_count in query.items():
            position = bisect_left(candidates.terms, term)
            if position < len(candidates.terms) and candidates.terms[position] == term:
                query_terms.append((term, query_count, candidates.inverse_frequencies[position]))
        ranked = []
        for document, length_adjustment in zip(candidates.documents, candidates.length_adjustments):
            skill = document.skill
            score = 0.0
            for term, query_count, inverse_frequency in query_terms:
                position = bisect_left(document.terms, term)
                if position == len(document.terms) or document.terms[position] != term:
                    continue
                count = document.counts[position]
                score += query_count * inverse_frequency * count * 2.2 / (count + length_adjustment)
            if state["turn"] == 0 and document.application_terms & {"first", "initial", "start", "beginning"}:
                score += 0.5
            if state["remaining_actions"] == 1 and document.application_terms & {"final", "finalizing", "output"}:
                score += 0.5
            if score > 0:
                ranked.append((score, skill))
        return tuple(skill for _, skill in sorted(
            ranked, key=lambda item: (-item[0], item[1].category, item[1].skill_id)
        ))

    def select(self, state, action_kind, original_action):
        """Rank candidate principles for the original task and action content.

        Ordinary prompt and observation templates do not enter text ranking.
        The teacher uses the unchanged interaction history to decide whether
        each candidate's original application conditions are satisfied.
        """
        self.validate_state(state)
        _text(action_kind, "action_kind")
        _text(original_action, "original_action")
        task = state["task"]
        if action_kind not in TASK_ACTIONS[task]:
            raise ValueError(f"Unknown action {action_kind} for task {task}")
        if self.max_skills == 0:
            return ()
        action_position = TASK_ACTIONS["searchqa"].index(action_kind)
        candidates = self._candidates[action_position]
        ranked = self._rank(candidates, state, original_action) if candidates.documents else ()
        return ranked[:self.max_skills]
