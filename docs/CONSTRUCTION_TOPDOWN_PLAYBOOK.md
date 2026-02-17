# Construction Top-Down Retrieval Playbook

## 1. Top-Down Hierarchy
Recommended retrieval hierarchy:

`portfolio -> program -> project -> site -> zone -> building -> level -> package -> work_type -> activity -> task`

Use this hierarchy as:
- Filter gate (hard constraints where known).
- Ranking boost (soft preference where partial context exists).
- Prompt grounding for rerank and answer generation.

## 2. Canonical Field Dictionary
Core identity and location:
- `portfolio_id`, `program_id`, `project_id`, `site_code`
- `asset_type`, `zone`, `building`, `level`

Execution structure:
- `package_code`, `work_type`, `discipline`, `activity`, `task_code`
- `wbs_code`, `csi_division`, `spec_section`

Delivery and control:
- `contractor`, `schedule_window`
- `safety_risk_class`, `quality_check_type`

Content blocks (recommended for lexical retrieval):
- `title`, `summary`, `content`
- `scope_text`, `method_statement`, `risk_register`
- `quality_checklist`, `schedule_notes`, `asset_tags`

## 3. Elasticsearch Mapping and Query
See template file:
- `infra/schemas/elasticsearch_documents_mapping.json`

Search design:
- `multi_match` on construction-aware boosted fields.
- top-down metadata fields mapped as exact keyword filters.
- optional recency and readiness scoring in `function_score`.

## 4. Milvus Vector Strategy
Collection:
- `id` (primary key)
- `embedding` (float vector)
- `title`, `content`
- `metadata` (json)

Index strategy (starting point):
- HNSW for lower-latency semantic retrieval.
- IVF_PQ for high-cardinality and memory pressure scenarios.

Partition strategy:
- Partition by `project_id` or `program_id` for large multi-project deployment.

## 5. Neo4j Graph Strategy
Nodes:
- `Document`, `WorkPackage`, `Task`, `Asset`, `Risk`, `QualityCheck`

Relationships:
- `(:Document)-[:BELONGS_TO]->(:WorkPackage)`
- `(:WorkPackage)-[:CONTAINS]->(:Task)`
- `(:Task)-[:AFFECTS]->(:Asset)`
- `(:Task)-[:HAS_RISK]->(:Risk)`
- `(:Task)-[:HAS_QC]->(:QualityCheck)`

Initialization template:
- `infra/schemas/neo4j_init.cypher`

## 6. Prompt Templates
Prompt templates should enforce deterministic output for machine use.

Rerank output contract:
- JSON object only: `{"ids":["doc1","doc2",...]}`

Priority:
1. Exact top-down path match.
2. WBS / CSI / spec alignment.
3. Safety / quality / schedule relevance.
4. Execution-readiness and recency.

Trade-specific guidance templates:
- `civil`: earthwork, excavation support, dewatering, settlement monitoring.
- `structure`: rebar/formwork/concrete sequencing, test and hold points.
- `mep`: route/interface clash, T&C readiness, shutdown window.
- `envelope`: lifting plan, anchor quality, weatherproofing, alignment.
- `safety`: permit-to-work, exclusion zone, critical risk control evidence.
- `general`: constructability, schedule feasibility, quality acceptance.

Code reference:
- `app/domain/construction.py`
  - `TRADE_PROMPT_TEMPLATES`
  - `infer_trade_from_query`
  - `build_construction_rerank_prompt`

Implementation:
- `app/domain/construction.py`

## 7. Scoring Policy
Baseline fusion:
- `final = w_elastic * rrf_elastic + w_vector * rrf_vector + w_graph * rrf_graph`

Elasticsearch stage score:
- lexical bool query + top-down should boosts
- function score additions:
  - top-down metadata exact-match weights
  - recency decay on `updated_at`
  - readiness factor via `metadata.execution_readiness`

Starting weights:
- `elastic=0.35`, `vector=0.45`, `graph=0.20`

Tuning guidance:
- Increase `elastic` for compliance/spec lookup traffic.
- Increase `vector` for natural language question traffic.
- Increase `graph` for dependency/path queries.

## 8. Evaluation Framework
Offline:
- nDCG@10, Recall@50, MRR@10 on labeled construction queries.
- Query buckets:
  - spec lookup
  - method statement lookup
  - risk/QC lookup
  - schedule sequencing lookup

Online:
- Zero-result rate
- Reformulation rate
- Click-through or acceptance proxy
- Time-to-first-useful-document

## 9. Rollout Steps
1. Lock schema and metadata contract.
2. Build 200-500 labeled construction queries.
3. Calibrate fusion weights and rerank prompt.
4. Launch shadow traffic.
5. Ramp with SLO and relevance guardrails.
