# Hybrid Search Platform — 아키텍처 및 알고리즘 보고서

이 문서는 README.md를 보완하여, **전체 시스템 아키텍처**, **데이터 흐름**, **핵심 알고리즘**을 한눈에 이해할 수 있도록 정리한 보고서입니다.

---

## 1. 시스템 개요

본 플랫폼은 **키워드(Elasticsearch) + 벡터(Milvus) + 그래프(Neo4j) + 로컬 소스**를 결합한 **하이브리드 검색**과 **RAG(Retrieval-Augmented Generation)**를 제공하는 프로덕션 지향 MVP입니다. 건설 도메인 top-down 컨텍스트(프로젝트/빌딩/레벨 등)를 지원하며, 선택적 LLM 리랭킹과 캐시(Redis)를 통해 대규모 동시 사용을 고려한 설계입니다.

---

## 2. 전체 아키텍처

### 2.1 고수준 구성도

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     FastAPI (app/main.py)               │
                    │  /v1/search, /v1/ask, /v1/upload, /v1/health, /ui ...   │
                    └───────────────────────────┬─────────────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │              Container (app/dependencies.py)            │
                    │  elastic, vector, graph, cache, embedding, reranker,    │
                    │  llm_gateway, uploads, sources, free_answer, hybrid     │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
    │                         HybridSearchService (app/services/hybrid_search_service.py)     │
    │  • merge_top_down_filters → cache key → (optional) cache get / lock                     │
    │  • embed(query) → ES / Milvus / Graph / Local 검색 병렬 실행 (asyncio.gather)             │
    │  • weighted_reciprocal_rank_fusion → (optional) LLM rerank → top_k → cache set          │
    └────────────────────────────────────────────┼────────────────────────────────────────────┘
                                                 │
    ┌──────────────┬──────────────┬──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌──────────────┐   ┌─────────────┐
│Redis   │   │Embedding │   │LLM      │   │Source   │   │Elasticsearch │   │Milvus       │
│Cache   │   │Service   │   │Reranker │   │Service  │   │Client        │   │VectorClient │
└────────┘   └──────────┘   └─────────┘   └─────────┘   └──────────────┘   └─────────────┘
                                                                                    │
                                                                                    ▼
                                                                            ┌─────────────┐
                                                                            │GraphSearch  │
                                                                            │Client(Neo4j)│
                                                                            └─────────────┘
```

### 2.2 디렉터리/레이어 구조

| 레이어 | 경로 | 역할 |
|--------|------|------|
| **API** | `app/main.py`, `app/api/routes/` | FastAPI 앱, 라우터(search, ui), 정적 파일 |
| **의존성** | `app/dependencies.py` | Container: 모든 클라이언트·서비스 생성 및 주입 |
| **서비스** | `app/services/` | HybridSearch, Cache, Embedding, LLMReranker, Source, Upload, FreeAnswer, SampleStore |
| **클라이언트** | `app/clients/` | Elasticsearch, Milvus, Graph(Neo4j) HTTP/드라이버 래퍼 |
| **도메인** | `app/domain/construction.py` | Top-down 필터, 쿼리 확장, ES 부스트/함수 스코어, 리랭크 프롬프트 |
| **모델** | `app/models/schemas.py` | Pydantic 스키마(SearchRequest, Candidate, AskRequest 등) |
| **설정** | `app/core/config.py` | pydantic-settings 기반 환경 변수 |
| **인프라** | `infra/` | ES 매핑, Neo4j Cypher, Redis 설정, Nginx |

---

## 3. 검색 파이프라인 (Search Flow)

### 3.1 POST /v1/search 흐름

1. **요청 정규화**  
   `SearchRequest`: query, top_k, use_cache, filters, top_down_context, rerank_with_llm, selected_source_ids, embedded_only.

2. **필터 병합**  
   `merge_top_down_filters(filters, top_down_context)` → 단일 `merged_filters` dict.  
   `selected_source_ids`는 정렬 후 캐시 키에 포함.

3. **캐시**  
   - `use_cache=True`이면 `cache.make_key({query, top_k, filters, selected_source_ids, embedded_only, rerank})`로 키 생성.  
   - `cache.get_json(key)` 히트 시 즉시 `SearchResponse(cache_hit=True)` 반환.  
   - 미스 시 `cache.acquire_lock(key)` → 락 획득 실패 시 `wait_for_value(key)`로 다른 요청이 채운 결과 대기.

4. **병렬 검색**  
   - `embedding = await embedding.embed(query)`  
   - 다음 네 가지를 `asyncio.gather`로 동시 실행:  
     - `elastic.search(query, top_k, merged_filters)`  
     - `vector.search(embedding, top_k, merged_filters, query_text=query)`  
     - `graph.search(query, top_k, merged_filters)`  
     - `source_service.local_search(query, top_k, ...)` (스레드 풀에서 실행)

5. **퓨전**  
   `weighted_reciprocal_rank_fusion(ranked_lists, weights)` → 소스별 가중 RRF 점수로 정렬된 단일 리스트.

6. **리랭크(선택)**  
   `rerank_with_llm=True`이면 `reranker.rerank(query, fused, top_k)` 호출 후 상위 top_k만 유지; 아니면 퓨전 결과를 그대로 top_k로 자르기.

7. **응답 및 캐시 저장**  
   `SearchResponse` 생성 후 `use_cache`이면 `cache.set_json(key, response)` 저장하고 락 해제.

### 3.2 POST /v1/ask 흐름 (RAG)

1. **검색**  
   `rerank_with_llm=False`로 `HybridSearchService.search()` 호출.

2. **히트 후처리**  
   - `_filter_hits`: selected_source_ids, embedded_only, top_down_context에 따른 필터링.  
   - 히트 수가 top_k 미만이면 `source_service.local_search`로 보강.  
   - 여전히 부족하면 `source_service.fallback_context_docs`로 컨텍스트 기준 문서 추가.  
   - 최종 `hits[:top_k]`.

3. **답변 생성**  
   - 히트 없음 → 고정 안내 문구 + `rag_synthesized=False`, `fallback_reason=no_matching_sources`.  
   - 있으면 상위 문서로 construction 도메인 프롬프트 구성 후 `llm_gateway.invoke(chat)` 호출.  
   - 성공 시 답변 텍스트 + `rag_synthesized=True`.  
   - LLM 실패/빈 출력 시 `_build_extract_answer`로 검색 결과 요약 반환 + `fallback_reason` 설정.

---

## 4. 핵심 알고리즘

### 4.1 Weighted Reciprocal Rank Fusion (RRF)

**위치**: `app/services/ranking.py`

**수식**:  
각 소스 `s`의 가중치 `w_s`와 상수 `k=60`에 대해, 문서 `d`의 퓨전 점수:

```
score(d) = Σ_s  w_s × 1/(k + rank_s(d))
```

- `rank_s(d)`: 소스 `s`에서 문서 `d`의 순위(1-based).  
- 한 소스에만 등장하는 문서도 점수가 있고, 여러 소스에 등장하면 점수가 누적됨.  
- 동일 문서는 한 번만 선택하고(`selected[cand.id]`), 최종 리스트는 `fused_score` 내림차순 정렬.

**설정**:  
`config`: `weight_elastic=0.35`, `weight_vector=0.45`, `weight_graph=0.20`, `weight_local=0.30`.  
소스별 리스트가 비어 있거나 가중치가 0이면 해당 소스는 기여하지 않음.

### 4.2 건설 도메인 쿼리 확장 (Elasticsearch용)

**위치**: `app/domain/construction.py` — `expand_construction_query`

- 쿼리 문자열에 사전 정의된 키워드(slab, pour, rebar, excavation, dewatering, facade, mep, qa, hse 등)가 포함되면 해당 동의어/연관어 목록을 쿼리 뒤에 이어 붙임.  
- 예: "slab" → "deck", "concrete slab", "pour" 등 추가.  
- ES는 `multi_match`로 원본 쿼리 + 확장된 텍스트를 함께 검색하여 recall 향상.

### 4.3 Elasticsearch 쿼리 구조

**위치**: `app/clients/elasticsearch_client.py`

- **Bool 쿼리**  
  - **must**: `multi_match` (query = 확장 쿼리, fields = `ELASTIC_QUERY_FIELDS`, type = best_fields).  
  - **should**: `build_context_should_clauses(filters)` — top-down 키(project_id, building, level 등)에 대한 term 부스트.  
  - **filter**: top_down 및 일반 필터를 `metadata.*` 또는 필드명에 매핑한 term 쿼리.

- **Function Score**  
  - **functions**: `build_function_score_functions(filters)`  
    - top_down 필터별 filter+weight.  
    - `updated_at`에 대한 gauss decay(최신 문서 선호).  
    - `metadata.execution_readiness`에 대한 field_value_factor.  
  - score_mode=sum, boost_mode=sum.

필드 부스트 예: title^4, summary^3, content^1.2, scope_text^2.5, method_statement^2.2 등 (`ELASTIC_QUERY_FIELDS`).

### 4.4 Milvus 벡터 검색

**위치**: `app/clients/milvus_client.py`

- **인덱스**: `embedding` 필드, AUTOINDEX, metric_type=IP(내적).  
- **필터**: `merged_filters`를 Milvus 표현식으로 변환(`key == "value"` 형태)하여 검색 시 적용.  
- **입력**: `embedding_service.embed(query)`로 얻은 1024차원 벡터.  
- **출력**: 각 히트의 `distance`(유사도)를 `raw_score`로 하는 `Candidate` 리스트.  
- 샘플 모드일 때는 ES/Graph와 마찬가지로 SampleStore의 키워드 검색으로 대체.

### 4.5 Neo4j 그래프(Fulltext) 검색

**위치**: `app/clients/graph_client.py`

- **인덱스**: `document_ft` fulltext (Document 노드의 title, content, summary, method_statement, risk_register).  
- **쿼리**: `CALL db.index.fulltext.queryNodes('document_ft', $q) YIELD node, score` 후, filters가 있으면 `WHERE node.key = $value` 형태로 필터링, `ORDER BY score DESC LIMIT $limit`.  
- 관계 기반 시그널은 현재 스키마에서는 fulltext 스코어만 사용하고, 노드 속성 필터로 top-down을 적용.

### 4.6 로컬 소스 검색 (SourceService)

**위치**: `app/services/source_service.py`

- **데이터**: `sample_data/sample_docs.json` + `data/uploaded_docs.json`을 합친 행 목록.  
- **알고리즘**:  
  - 쿼리/문서를 `TOKEN_RE`로 토큰화(영문·숫자·한글 등).  
  - selected_source_ids, embedded_only, top_down 필터로 후보 필터링.  
  - 쿼리 토큰과 제목·본문 토큰의 교집합 크기 + 제목 매칭 가중(2배)으로 스코어.  
  - 선택된 소스면 보너스 점수.  
- **fallback_context_docs**: 검색어 매칭 없이, 선택/업로드/업로드 시각 등으로 정렬해 컨텍스트 문서를 보충할 때 사용.

### 4.7 LLM 리랭킹

**위치**: `app/services/llm_reranker.py`, `app/domain/construction.py` — `build_construction_rerank_prompt`

- **입력**: 쿼리 + 퓨전 후 상위 50개 Candidate.  
- **프롬프트**:  
  - top-down 우선순위(프로젝트/사이트/빌딩/레벨/패키지/작업 등), WBS/CSI/spec, 안전/품질/일정 의도, 최신·실행 준비도.  
  - `infer_trade_from_query`로 civil/structure/mep/envelope/safety/general 중 트레이드 추론 후 해당 `TRADE_PROMPT_TEMPLATES` 문구 삽입.  
  - 출력 형식: `{"ids": ["doc_id_1", "doc_id_2", ...]}`.  
- **동작**: LLM 응답 JSON에서 `ids` 순서대로 Candidate를 재정렬하고 상위 top_k만 반환. 파싱/호출 실패 시 원본 순서 유지.

### 4.8 임베딩

**위치**: `app/services/embedding_service.py`

- **Provider**: openai, bedrock, ollama.  
- **Ollama**: `litellm.aembedding`으로 `ollama_base_url` / `ollama_base_urls` 순서로 시도, 사용 가능한 모델 목록 조회 후 후보 모델로 임베딩 요청.  
- **Fallback**: API 불가 시 SHA256 기반 결정론적 의사 벡터 생성(동일 텍스트 → 동일 벡터).  
- **차원**: `vector_dim=1024`, 부족하면 0 패딩 또는 자르기.

### 4.9 캐시 및 동시성

**위치**: `app/services/cache.py`

- **키**: 요청 파라미터(쿼리, top_k, filters, selected_source_ids, embedded_only, rerank)의 JSON을 sort_keys로 직렬화 후 SHA256 해시, `search:{hash}`.  
- **저장**: Redis `set key value ex=TTL`.  
- **Stampede 방지**: 캐시 미스 시 `lock:{key}`로 NX 락 획득, TTL(기본 8초). 실패 시 `wait_for_value`로 짧은 간격 폴링(약 1.5초 상한).  
- **네임스페이스**: `REDIS_NAMESPACE`로 키 접두어 분리.

---

## 5. 업로드 및 인덱싱

**위치**: `app/services/upload_document_service.py`

1. **파일 수신** → 파일명 정규화/인코딩 보정, `data/uploads/`에 저장.  
2. **파서 선택**: 확장자/Content-Type으로 text, pdf, docx, xlsx, pptx, html, hwpx, hwp 등 결정.  
3. **텍스트 추출**: 네이티브(openpyxl, docx, pptx, fitz, olefile 등) 시도 후, 실패 시 **Tika** HTTP(form/put/rmeta) 호출.  
4. **문서 객체 구성**: id=`upl-{uuid}`, title, content, summary, metadata(top_down_context, uploaded_at, embedded, parser 등).  
5. **다중 백엔드 인덱싱**:  
   - Elasticsearch: `index_document(doc)`  
   - Milvus: `embed(content)` 후 `index_document(doc, embedding)`  
   - Neo4j: `index_document(doc)` (MERGE Document, fulltext 인덱스 활용)  
6. **로컬 인덱스**: `data/uploaded_docs.json`에 문서 append.  
7. **응답**: file_name, doc_id, indexed, parser, indexed_backends, extracted, warning.

---

## 6. 샘플 모드 (Sample Mode)

- **설정**: `sample_mode=True`일 때 Elasticsearch/Milvus/Graph 클라이언트는 실제 서버 대신 `SampleStore`를 사용.  
- **SampleStore**: `sample_data/sample_docs.json` + `uploads_data_path` JSON을 읽어, 토큰 기반 키워드 검색(keyword_search) 및 그래프 스타일 검색(graph_search)을 메모리에서 수행.  
- **용도**: 외부 인프라 없이 로직·API·UI 검증.

---

## 7. 설정 요약 (알고리즘 관련)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| weight_elastic | 0.35 | RRF에서 키워드 소스 가중치 |
| weight_vector | 0.45 | RRF에서 벡터 소스 가중치 |
| weight_graph | 0.20 | RRF에서 그래프 소스 가중치 |
| weight_local | 0.30 | RRF에서 로컬 소스 가중치 |
| max_candidates_per_source | 100 | 소스별 최대 후보 수 |
| vector_dim | 1024 | 임베딩 차원 |
| redis_ttl_sec | 300 | 검색 캐시 TTL |
| redis_lock_ttl_sec | 8 | 캐시 락 TTL |
| redis_lock_wait_ms | 1500 | 락 대기 최대 시간 |

---

## 8. 데이터 모델 요약

- **Candidate**: id, title, content, source(elastic|vector|graph|local|upload|sample), raw_score, fused_score, metadata.  
- **ConstructionTopDownContext**: project_id, building, level, package_code, work_type, task_code, wbs_code 등 (schemas + domain TOP_DOWN_FILTER_KEYS).  
- **SearchRequest / AskRequest**: query, top_k, filters, top_down_context, selected_source_ids, embedded_only, use_cache, (Ask) provider, model, max_tokens.

---

이 문서는 코드베이스와 설정을 기준으로 작성되었으며, README.md의 Quick Start, API 엔드포인트, UI, 배포 가이드와 함께 사용하면 전체 플랫폼을 빠르게 파악할 수 있습니다.

---

## 10. 2026-02 Update Notes

### 10.1 Source Delete Reliability

- Endpoint `POST /v1/sources/delete` now supports:
  - `source_ids: []`
  - `delete_all_uploads: bool` (delete all deletable uploaded sources without sending large id arrays)
  - `skip_backend_cleanup: bool` (optional fast delete mode)
- Backend cleanup (`elastic/vector/graph`) now checks backend health first and skips down backends.
- Cleanup calls use bounded timeout and concurrency to avoid long blocking on large delete requests.

### 10.2 Upload Parser Chain (Tika + DPE Fallback)

- Parser strategy was updated to:
  1. Tika-first for advanced document types
  2. Upstage DPE fallback when extracted text is too short or failed
  3. Native parser fallback as last step
- DPE integration is optional and non-blocking:
  - If DPE is disabled/unconfigured/unreachable, upload continues and DPE is skipped.
- After text extraction, upload service performs automatic top-down context extraction:
  - Rule-based inference from filename/content (`project_id`, `building`, `level`, `package_code`, `work_type`, etc.)
  - Optional LLM refinement (configurable)
  - Manual upload form fields override inferred values.
- Upload always attempts vector embedding/indexing; success is exposed as `embedding_indexed` in upload response.
- New settings:
  - `UPSTAGE_DPE_ENABLED`
  - `UPSTAGE_DPE_API_KEY`
  - `UPSTAGE_DPE_BASE_URL`
  - `UPSTAGE_DPE_TIMEOUT_SEC`
- `UPSTAGE_DPE_MIN_CHARS`
- `AUTO_CONTEXT_EXTRACT_ENABLED`
- `AUTO_CONTEXT_EXTRACT_USE_LLM`
- `AUTO_CONTEXT_EXTRACT_MAX_CHARS`
- `AUTO_CONTEXT_EXTRACT_MIN_CHARS_FOR_LLM`
- `AUTO_CONTEXT_EXTRACT_LLM_MODEL`

### 10.3 Context Auto-Extraction and Upload Embedding

- Upload service now infers top-down fields from filename/content using rule-based extraction.
- Optional LLM refinement can be enabled via config (`AUTO_CONTEXT_EXTRACT_USE_LLM=true`).
- Manual upload context remains highest priority and overrides inferred values.
- Upload flow always attempts vector embedding/indexing and reports `embedding_indexed` in response.
- Upload response includes `automatic_context_extracted` and merged `top_down_context` used for indexing.
