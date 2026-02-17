from __future__ import annotations

from app.domain.construction import (
    build_construction_rerank_prompt,
    build_context_should_clauses,
    build_function_score_functions,
    expand_construction_query,
    infer_trade_from_query,
)


def test_expand_construction_query_adds_synonyms():
    q = "slab pour sequence"
    expanded = expand_construction_query(q)
    assert "deck" in expanded
    assert "placement" in expanded


def test_build_context_should_clauses_contains_metadata_terms():
    clauses = build_context_should_clauses(
        {
            "project_id": "PROJ-1",
            "package_code": "PKG-1",
            "unknown_key": "x",
        }
    )
    assert any("metadata.project_id" in c["term"] for c in clauses)
    assert any("metadata.package_code" in c["term"] for c in clauses)


def test_build_function_score_functions_contains_core_signals():
    fns = build_function_score_functions({"project_id": "P1", "package_code": "PKG-1"})
    assert any("filter" in f and "weight" in f for f in fns)
    assert any("gauss" in f for f in fns)
    assert any("field_value_factor" in f for f in fns)


def test_trade_prompt_is_included():
    trade = infer_trade_from_query("slab pour sequence for rebar")
    assert trade == "structure"
    prompt = build_construction_rerank_prompt("slab pour sequence for rebar", [])
    assert "TradeFocus: structure" in prompt
