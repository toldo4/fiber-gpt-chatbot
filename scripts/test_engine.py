"""
Offline regression tests for the decision tree engine. No API keys, no network.

Every test in the first group corresponds to a way the previous engine produced
a confident answer it had not earned.

Usage:
    python3 test_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from decision_tree_engine import (
    DecisionTree,
    DecisionTreeEngine,
    Question,
    TreeError,
)
from extractors import Extraction, ExtractionRequest, ExtractionError, parse_json_object, finalize
from rag_index import Chunk, Hit, tokenize
from values import coerce_boolean, coerce_categorical, coerce_number

TREE_YAML = Path(__file__).parent / "decision_tree.yaml"

_failures: list[str] = []
_passes = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passes
    if condition:
        _passes += 1
        print(f"  ok   {name}")
    else:
        _failures.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")


def section(title: str) -> None:
    print(f"\n{title}")


# ── Stub extractor ────────────────────────────────────────────────────────────

class StubExtractor:
    """Returns scripted answers so traversal can be tested without a model."""

    def __init__(self, answers: dict[str, tuple], default: tuple = ("unknown", 0.0)) -> None:
        self.answers = answers
        self.default = default
        self.calls: list[str] = []

    def extract(self, request: ExtractionRequest) -> Extraction:
        self.calls.append(request.key)
        value, confidence = self.answers.get(request.key, self.default)
        return Extraction(value=value, confidence=confidence, reasoning="stub")


def load_engine(extractor, **kwargs) -> DecisionTreeEngine:
    return DecisionTreeEngine(DecisionTree.from_yaml(TREE_YAML), extractor, **kwargs)


# ── 1. The bugs that produced the bad run ─────────────────────────────────────

def test_unknowns_are_reported_not_hidden() -> None:
    """The old engine answered AOAC 2001.03 with 4 of 5 attributes unknown."""
    section("1. Unknowns surface instead of becoming a confident answer")

    extractor = StubExtractor({"requires_cooking": ("yes", 0.9)})
    result = load_engine(extractor).run(
        "canned kidney beans, requires heating before eating, "
        "no added NDO, high resistant starch content"
    )

    check("result carries a warning", result.warning, f"warning={result.warning}")
    check("result is not marked reliable", not result.is_reliable)
    check("overall confidence is zero", result.confidence == 0.0, f"{result.confidence}")
    check(
        "unresolved names every unanswered attribute",
        set(result.unresolved) >= {
            "targeted_methods_measurable",
            "fibre_content_g_per_100g",
            "contains_ndo",
            "ndo_in_fibre_definition",
            "needs_idf_sdf_separate",
        },
        str(result.unresolved),
    )
    check(
        "the guessed branches are recorded as assumptions",
        len(result.assumptions) >= 2,
        str(result.assumptions),
    )


def test_no_edge_falls_through_to_yaml_order() -> None:
    """`unknown` must never resolve to whichever branch was typed first."""
    section("2. No silent fall-through to the first YAML key")

    data = yaml.safe_load(TREE_YAML.read_text())
    missing = [
        node_id
        for node_id, node in data["nodes"].items()
        if "unknown" not in node["edges"]
    ]
    check("every node declares an unknown edge", not missing, str(missing))

    # And a tree without one is rejected at load, not at 3am in production.
    broken = yaml.safe_load(TREE_YAML.read_text())
    del broken["nodes"]["idf_sdf_cooked_ndo"]["edges"]["unknown"]
    errors = DecisionTree(broken).validate()
    check(
        "a tree missing an unknown edge fails validation",
        any("unknown" in e for e in errors),
        str(errors),
    )
    try:
        DecisionTreeEngine(DecisionTree(broken), StubExtractor({}))
        check("the engine refuses to construct on an invalid tree", False, "no raise")
    except TreeError:
        check("the engine refuses to construct on an invalid tree", True)


def test_unknown_fibre_does_not_skip_the_low_fibre_guard_silently() -> None:
    section("3. Unknown fibre content is flagged, not assumed above threshold")

    extractor = StubExtractor({}, default=("unknown", 0.0))
    result = load_engine(extractor).run("some product")
    check(
        "the unknown-fibre assumption is stated",
        any("0.5" in a for a in result.assumptions),
        str(result.assumptions),
    )

    low = StubExtractor({
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (0.2, 0.9),
    })
    stopped = load_engine(low).run("fresh apple juice, 0.2 g/100g fibre")
    check("low fibre stops the traversal", stopped.warning and stopped.can_override)
    check(
        "the warning message is the one from the YAML, not hardcoded",
        "limit of quantification" in stopped.warning_message,
        stopped.warning_message[:80],
    )


def test_ordering_of_yaml_keys_does_not_change_the_answer() -> None:
    section("4. Reordering edges in the YAML cannot change a result")

    answers = {
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (6.0, 0.9),
        "requires_cooking": ("yes", 0.9),
        "contains_ndo": ("no", 0.9),
    }
    baseline = load_engine(StubExtractor(answers)).run("kidney beans")

    shuffled = yaml.safe_load(TREE_YAML.read_text())
    for node in shuffled["nodes"].values():
        node["edges"] = dict(reversed(list(node["edges"].items())))
    reordered = DecisionTreeEngine(DecisionTree(shuffled), StubExtractor(answers)).run(
        "kidney beans"
    )

    check(
        "same method with edges reversed",
        baseline.method == reordered.method,
        f"{baseline.method!r} vs {reordered.method!r}",
    )
    check("same path with edges reversed", baseline.path == reordered.path)


def test_low_confidence_is_treated_as_unknown() -> None:
    section("5. A low-confidence answer is not trusted")

    confident = load_engine(StubExtractor({
        "targeted_methods_measurable": ("yes", 0.9),
    })).run("x")
    hedged = load_engine(StubExtractor({
        "targeted_methods_measurable": ("yes", 0.2),
    })).run("x")

    check("high confidence takes the yes branch", confident.path == ["targeted_methods"])
    check("low confidence does not", hedged.path != ["targeted_methods"], str(hedged.path))
    check(
        "the hedged attribute is unresolved",
        "targeted_methods_measurable" in hedged.unresolved,
    )


# ── 2. Asking the user ────────────────────────────────────────────────────────

def test_user_attributes_are_asked_not_guessed() -> None:
    section("6. User-owned attributes are asked, never inferred")

    extractor = StubExtractor({
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (6.0, 0.9),
        "requires_cooking": ("yes", 0.9),
        "contains_ndo": ("no", 0.9),
    })
    session = load_engine(extractor).start("canned kidney beans")

    check("traversal pauses on a question", isinstance(session.pending, Question))
    check(
        "the question is one only the user can answer",
        session.pending.attribute == "needs_idf_sdf_separate",
        str(session.pending.attribute),
    )
    check("the question explains what it changes", bool(session.pending.why))
    check(
        "no extractor call was made for a user attribute",
        "needs_idf_sdf_separate" not in extractor.calls,
        str(extractor.calls),
    )

    session.answer("yes")
    check("answering completes the traversal", session.done)
    check(
        "the answer decides the outcome",
        session.result.method == "AOAC 991.43 (expected to increase variability)",
        session.result.method,
    )
    check("no unresolved attributes remain", not session.result.unresolved,
          str(session.result.unresolved))
    check("confidence is high", session.result.confidence >= 0.9)


def test_answering_does_not_re_extract() -> None:
    section("7. Re-walking after an answer costs no extra model calls")

    extractor = StubExtractor({
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (6.0, 0.9),
        "requires_cooking": ("yes", 0.9),
        "contains_ndo": ("yes", 0.9),
    })
    session = load_engine(extractor).start("inulin-enriched bread")
    before = list(extractor.calls)
    session.answer("yes")           # ndo_in_fibre_definition
    session.answer("no")            # needs_idf_sdf_separate
    check("no attribute was extracted twice", len(set(extractor.calls)) == len(extractor.calls),
          str(extractor.calls))
    check("no new extractions after the first pause", extractor.calls == before,
          f"{before} → {extractor.calls}")
    check("reached the NDO outcome", session.result.method == "AOAC 2001.03",
          session.result.method)


def test_declining_to_answer_does_not_loop() -> None:
    section("8. 'I don't know' is accepted and not re-asked")

    session = load_engine(StubExtractor({
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (6.0, 0.9),
        "requires_cooking": ("yes", 0.9),
        "contains_ndo": ("no", 0.9),
    })).start("kidney beans")

    asked: list[str] = []
    for _ in range(10):
        if session.done:
            break
        asked.append(session.pending.attribute)
        session.answer("unknown")

    check("traversal terminates", session.done, str(asked))
    check("nothing was asked twice", len(asked) == len(set(asked)), str(asked))
    check("the unresolved answer is carried into the result",
          "needs_idf_sdf_separate" in session.result.unresolved)
    check("the outcome warns about it", session.result.warning)


def test_low_confidence_extraction_escalates_to_the_user() -> None:
    section("9. A hedged model answer becomes a question")

    session = load_engine(StubExtractor({
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (6.0, 0.3),
    })).start("mystery sample")

    check("the hedged attribute is put to the user",
          session.pending is not None
          and session.pending.attribute == "fibre_content_g_per_100g",
          str(session.pending))
    check("the question records the weak extraction it came from",
          session.pending.because_low_confidence is not None)

    session.answer("6 g/100 g")
    check("a messy human answer is parsed",
          session.attrs["fibre_content_g_per_100g"].value == 6.0,
          str(session.attrs["fibre_content_g_per_100g"].value))


def test_override_continues_past_the_stop() -> None:
    section("10. can_override is wired up, not decorative")

    session = load_engine(StubExtractor({
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (0.2, 0.95),
        "requires_cooking": ("no", 0.9),
        "contains_ndo": ("no", 0.9),
        "contains_resistant_starch": ("no", 0.9),
    })).start("apple juice", known={"needs_idf_sdf_separate": "no"})

    check("stops at the low-fibre outcome", session.result.can_override)
    session.override()
    check("override continues the traversal",
          session.result.method.startswith("Any of the following"), session.result.method)
    check("the override is recorded as an assumption",
          any("at the user's request" in a for a in session.result.assumptions),
          str(session.result.assumptions))


# ── 3. Composite NDO node ─────────────────────────────────────────────────────

def test_ndo_conjunction_is_two_separate_questions() -> None:
    section("11. The NDO node splits product fact from regulatory fact")

    base = {
        "targeted_methods_measurable": ("no", 0.9),
        "fibre_content_g_per_100g": (6.0, 0.9),
        "requires_cooking": ("yes", 0.9),
    }

    # Product has no NDO → the regulatory half never needs to be asked.
    engine = load_engine(StubExtractor({**base, "contains_ndo": ("no", 0.9)}))
    session = engine.start("plain kidney beans")
    check("a 'no' on the product half short-circuits",
          session.pending.attribute == "needs_idf_sdf_separate",
          str(session.pending.attribute))

    # Product has NDO → the jurisdiction question becomes relevant.
    engine = load_engine(StubExtractor({**base, "contains_ndo": ("yes", 0.9)}))
    session = engine.start("bread with added inulin")
    check("a 'yes' on the product half raises the regulatory half",
          session.pending.attribute == "ndo_in_fibre_definition",
          str(session.pending.attribute))

    session.answer("no")
    session.answer("no")
    check("NDO present but excluded routes to the non-NDO branch",
          session.result.path[-1] == "idf_sdf_cooked_no_ndo", str(session.result.path))


# ── 4. Structural validation ──────────────────────────────────────────────────

def test_validation_catches_structural_faults() -> None:
    section("12. Validation catches broken trees")

    def errors_for(mutate) -> list[str]:
        data = yaml.safe_load(TREE_YAML.read_text())
        mutate(data)
        return DecisionTree(data).validate()

    def make_cycle(data):
        data["nodes"]["idf_sdf_cooked_ndo"]["edges"]["no"] = {
            "type": "node", "next": "targeted_methods"
        }

    def make_unreachable(data):
        data["nodes"]["orphan"] = {
            "attribute": "requires_cooking",
            "edges": {
                "yes": {"type": "outcome", "result": "x"},
                "no": {"type": "outcome", "result": "y"},
                "unknown": {"type": "outcome", "result": "z"},
            },
        }

    check("cycle detected",
          any("cycle" in e for e in errors_for(make_cycle)))
    check("unreachable node detected",
          any("unreachable" in e for e in errors_for(make_unreachable)))
    check("dangling next detected",
          any("not a node" in e for e in errors_for(
              lambda d: d["nodes"]["requires_cooking"]["edges"]["yes"].update(next="nope"))))
    check("undefined attribute detected",
          any("undefined attribute" in e for e in errors_for(
              lambda d: d["nodes"]["rs_check"].update(attribute="nope"))))
    check("missing threshold detected",
          any("threshold" in e for e in errors_for(
              lambda d: d["nodes"]["fibre_content"].pop("threshold"))))
    check("warning without a message detected",
          any("warning_message" in e for e in errors_for(
              lambda d: d["nodes"]["fibre_content"]["edges"]["below"].pop("warning_message"))))
    check("override without a target detected",
          any("override_next" in e for e in errors_for(
              lambda d: d["nodes"]["fibre_content"]["edges"]["below"].pop("override_next"))))
    check("attribute missing extraction_prompt detected",
          any("extraction_prompt" in e for e in errors_for(
              lambda d: d["attributes"]["requires_cooking"].pop("extraction_prompt"))))


# ── 5. Value coercion ─────────────────────────────────────────────────────────

def test_value_coercion() -> None:
    section("13. Messy values are coerced, not crashed on")

    cases = [
        ("6", 6.0), ("~6", 6.0), ("6 g/100g", 6.0), ("6.5", 6.5), ("6,5", 6.5),
        ("5-7", 6.0), ("5 to 7", 6.0), ("about 0.2 g per 100 g", 0.2),
        (None, None), ("null", None), ("unknown", None), ("no idea", None), (6, 6.0),
    ]
    bad = [(raw, coerce_number(raw)) for raw, want in cases if coerce_number(raw) != want]
    check("numbers parse out of messy text", not bad, str(bad))

    bools = [("Yes.", "yes"), ("NO", "no"), (True, "yes"), (False, "no"),
             ("true", "yes"), ("maybe", "unknown"), (None, "unknown"), ("y", "yes")]
    bad_bools = [(raw, coerce_boolean(raw)) for raw, want in bools
                 if coerce_boolean(raw) != want]
    check("booleans normalise", not bad_bools, str(bad_bools))

    check("categorical snaps case-insensitively",
          coerce_categorical("EU", ["eu", "us"]) == "eu")
    check("unrecognised categorical is unknown",
          coerce_categorical("mars", ["eu", "us"]) == "unknown")


# ── 6. Extractor plumbing ─────────────────────────────────────────────────────

def test_json_parsing_handles_nesting_and_fences() -> None:
    section("14. Model output parsing")

    nested = 'Sure!\n{"value": "yes", "meta": {"a": 1}, "confidence": 0.8}\n'
    check("nested objects survive (the old non-greedy regex did not)",
          parse_json_object(nested)["meta"] == {"a": 1})

    fenced = '```json\n{"value": "no", "confidence": 0.7}\n```'
    check("markdown fences are stripped", parse_json_object(fenced)["value"] == "no")

    try:
        parse_json_object("I cannot answer that.")
        check("unparseable output raises rather than becoming unknown", False, "no raise")
    except ExtractionError:
        check("unparseable output raises rather than becoming unknown", True)


def test_finalize_validates_and_cites() -> None:
    section("15. Extraction finalisation")

    hits = [
        Hit(Chunk("1", "017.pdf", "Methods", "Beans are high in resistant starch."), 0.9, 0, 0),
        Hit(Chunk("2", "025.pdf", "Results", "Inulin is an NDO."), 0.8, 1, None),
    ]
    request = ExtractionRequest(
        key="contains_resistant_starch", question="RS?", type="boolean",
        instructions="?", product_description="beans", hits=hits,
    )

    got = finalize(request, {"value": "Yes", "confidence": 0.82,
                             "reasoning": "legume", "sources": [1]})
    check("value is normalised", got.value == "yes")
    check("evidence resolves to the cited chunk",
          [str(e) for e in got.evidence] == ["017.pdf — Methods"], str(got.evidence))

    out_of_range = finalize(request, {"value": "yes", "confidence": 5,
                                      "sources": [9, "x", 2]})
    check("confidence is clamped", out_of_range.confidence == 1.0)
    check("out-of-range citations are dropped",
          [str(e) for e in out_of_range.evidence] == ["025.pdf — Results"],
          str(out_of_range.evidence))

    unknown = finalize(request, {"value": "unknown", "confidence": 0.99})
    check("an unknown value cannot carry confidence", unknown.confidence == 0.0)
    check("an unknown value is not trusted", not unknown.trusted(0.5))


def test_tokenizer_keeps_method_codes_whole() -> None:
    section("16. Lexical tokenizer")

    tokens = tokenize("Compare AOAC 991.43 with AOAC 2017.16 and 2001.03.")
    check("method codes stay intact",
          {"991.43", "2017.16", "2001.03"} <= set(tokens), str(tokens))
    check("words are lowercased", "aoac" in tokens, str(tokens))


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_unknowns_are_reported_not_hidden,
    test_no_edge_falls_through_to_yaml_order,
    test_unknown_fibre_does_not_skip_the_low_fibre_guard_silently,
    test_ordering_of_yaml_keys_does_not_change_the_answer,
    test_low_confidence_is_treated_as_unknown,
    test_user_attributes_are_asked_not_guessed,
    test_answering_does_not_re_extract,
    test_declining_to_answer_does_not_loop,
    test_low_confidence_extraction_escalates_to_the_user,
    test_override_continues_past_the_stop,
    test_ndo_conjunction_is_two_separate_questions,
    test_validation_catches_structural_faults,
    test_value_coercion,
    test_json_parsing_handles_nesting_and_fences,
    test_finalize_validates_and_cites,
    test_tokenizer_keeps_method_codes_whole,
]

if __name__ == "__main__":
    for test in ALL_TESTS:
        test()

    print(f"\n{'─' * 60}")
    if _failures:
        print(f"{len(_failures)} failed, {_passes} passed\n")
        for failure in _failures:
            print(f"  FAIL {failure}")
        sys.exit(1)
    print(f"all {_passes} checks passed")
