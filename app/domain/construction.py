from __future__ import annotations

import json
from typing import Dict, List, Optional, Union


TOP_DOWN_LEVELS: List[str] = [
    "portfolio",
    "program",
    "project",
    "site",
    "zone",
    "building",
    "level",
    "package",
    "work_type",
    "activity",
    "task",
]


ELASTIC_QUERY_FIELDS: List[str] = [
    "title^4",
    "summary^3",
    "content^1.2",
    "scope_text^2.5",
    "method_statement^2.2",
    "risk_register^1.8",
    "quality_checklist^1.6",
    "schedule_notes^1.6",
    "wbs_code^3.2",
    "package_code^3.0",
    "task_code^3.0",
    "csi_division^2.0",
    "spec_section^2.0",
    "asset_tags^1.6",
]


TOP_DOWN_FILTER_KEYS: List[str] = [
    "portfolio_id",
    "program_id",
    "project_id",
    "site_code",
    "asset_type",
    "zone",
    "building",
    "level",
    "package_code",
    "work_type",
    "discipline",
    "activity",
    "task_code",
    "wbs_code",
    "csi_division",
    "spec_section",
    "contractor",
    "safety_risk_class",
    "quality_check_type",
    "schedule_window",
]

CONSTRUCTION_SYNONYMS: Dict[str, List[str]] = {
    "slab": ["deck", "concrete slab", "pour"],
    "pour": ["placement", "casting", "concrete pour"],
    "rebar": ["reinforcement", "steel bar", "rebars"],
    "excavation": ["earthwork", "digging", "cut"],
    "dewatering": ["groundwater control", "well point"],
    "facade": ["envelope", "curtain wall", "cladding"],
    "mep": ["mechanical electrical plumbing", "hvac", "fire protection"],
    "qa": ["quality assurance", "quality check", "inspection"],
    "hse": ["safety", "ehs", "risk control"],
}

TRADE_PROMPT_TEMPLATES: Dict[str, str] = {
    "civil": (
        "Focus on earthwork, excavation support, dewatering, temporary works, and settlement control. "
        "Prioritize geotechnical risk and monitoring thresholds."
    ),
    "structure": (
        "Focus on reinforcement, formwork, concrete sequence, curing, and tolerance controls. "
        "Prioritize hold points, test records, and pour readiness."
    ),
    "mep": (
        "Focus on routing, interface clashes, testing and commissioning, and access constraints. "
        "Prioritize coordination drawings and shutdown windows."
    ),
    "envelope": (
        "Focus on facade anchors, lifting plans, weatherproofing, and plumb/alignment quality. "
        "Prioritize installation sequencing and temporary fixation checks."
    ),
    "safety": (
        "Focus on high-risk activities, permits, exclusion zones, and emergency response controls. "
        "Prioritize critical risk controls and verification evidence."
    ),
    "general": (
        "Focus on constructability, schedule feasibility, quality acceptance criteria, and risk controls."
    ),
}


def merge_top_down_filters(
    filters: Optional[Dict[str, Union[str, int, float, bool]]],
    context: Optional[Dict[str, Union[str, int, float, bool]]],
) -> Dict[str, Union[str, int, float, bool]]:
    out: Dict[str, Union[str, int, float, bool]] = {}
    if filters:
        out.update(filters)
    if context:
        for key, value in context.items():
            if value is None:
                continue
            out[key] = value
    return out


def expand_construction_query(query: str) -> str:
    q = query.lower()
    expansions: List[str] = []
    for key, values in CONSTRUCTION_SYNONYMS.items():
        if key in q:
            expansions.extend(values)
    if not expansions:
        return query
    expanded = " ".join(expansions)
    return f"{query} {expanded}"


def build_context_should_clauses(filters: Dict[str, Union[str, int, float, bool]]) -> List[Dict]:
    clauses: List[Dict] = []
    boost_by_key: Dict[str, float] = {
        "project_id": 4.0,
        "site_code": 3.5,
        "zone": 3.0,
        "building": 3.0,
        "level": 2.5,
        "package_code": 4.0,
        "work_type": 3.0,
        "task_code": 3.5,
        "wbs_code": 3.2,
        "csi_division": 2.2,
        "spec_section": 2.2,
    }
    for key, value in filters.items():
        if key not in TOP_DOWN_FILTER_KEYS:
            continue
        clauses.append(
            {
                "term": {
                    f"metadata.{key}": {
                        "value": value,
                        "boost": boost_by_key.get(key, 1.0),
                    }
                }
            }
        )
    return clauses


def build_function_score_functions(filters: Dict[str, Union[str, int, float, bool]]) -> List[Dict]:
    functions: List[Dict] = []
    weight_by_key: Dict[str, float] = {
        "project_id": 10.0,
        "site_code": 7.0,
        "building": 6.0,
        "level": 4.0,
        "package_code": 9.0,
        "work_type": 7.0,
        "task_code": 8.0,
        "wbs_code": 6.0,
        "spec_section": 5.0,
    }
    for key, value in filters.items():
        if key not in TOP_DOWN_FILTER_KEYS:
            continue
        functions.append(
            {
                "filter": {"term": {f"metadata.{key}": value}},
                "weight": weight_by_key.get(key, 2.0),
            }
        )

    functions.append(
        {
            "gauss": {
                "updated_at": {
                    "origin": "now",
                    "scale": "365d",
                    "decay": 0.5,
                }
            },
            "weight": 0.3,
        }
    )
    functions.append(
        {
            "field_value_factor": {
                "field": "metadata.execution_readiness",
                "factor": 0.2,
                "missing": 1.0,
                "modifier": "sqrt",
            }
        }
    )
    return functions


def infer_trade_from_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["excavation", "earthwork", "dewatering", "slope"]):
        return "civil"
    if any(k in q for k in ["rebar", "concrete", "slab", "formwork", "pour"]):
        return "structure"
    if any(k in q for k in ["mep", "hvac", "sprinkler", "commissioning", "duct"]):
        return "mep"
    if any(k in q for k in ["facade", "curtain wall", "cladding", "envelope"]):
        return "envelope"
    if any(k in q for k in ["safety", "permit", "risk", "hse", "ehs"]):
        return "safety"
    return "general"


def build_construction_rerank_prompt(query: str, docs: List[Dict]) -> str:
    docs_json = json.dumps(docs, ensure_ascii=True)
    trade = infer_trade_from_query(query)
    trade_guidance = TRADE_PROMPT_TEMPLATES.get(trade, TRADE_PROMPT_TEMPLATES["general"])
    return (
        "You are a construction domain retrieval reranker.\n"
        "Use top-down context priority:\n"
        "1) Exact match on project/site/zone/building/level/package/work_type/task.\n"
        "2) Match on WBS, CSI division, and specification section.\n"
        "3) Relevance to safety, quality, schedule, and constructability intent.\n"
        "4) Prefer recent and execution-ready documents when ties exist.\n"
        f"TradeFocus: {trade}\n"
        f"TradeGuidance: {trade_guidance}\n"
        "Return strict JSON object: {\"ids\": [\"doc_id_1\", \"doc_id_2\", ...]}.\n"
        f"Query: {query}\n"
        f"TopDownLevels: {TOP_DOWN_LEVELS}\n"
        f"Documents: {docs_json}"
    )
