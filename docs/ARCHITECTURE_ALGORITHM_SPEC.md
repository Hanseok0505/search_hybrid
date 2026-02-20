# 아키텍처 및 알고리즘 설명서

현재 코드 기준으로 **전체 아키텍처**, **파일/모듈 역할**, **핵심 알고리즘**을 정리한 설명서입니다. 코드 수정 없이 문서만 제공합니다.

---

## 1. 시스템 개요

| 항목 | 설명 |
|------|------|
| **목적** | 키워드 + 벡터 + 그래프 + 로컬 소스를 결합한 하이브리드 검색 및 RAG(Retrieval-Augmented Generation) |
| **스택** | FastAPI, Elasticsearch, Milvus, Neo4j, Redis, Tika(문서 파싱) |
| **도메인** | 건설 top-down 컨텍스트(프로젝트/빌딩/레벨/패키지/작업 등) |
| **선택 기능** | LLM 리랭킹, 검색 캐시(Redis), Ollama/OpenAI/Bedrock 등 LLM 연동 |

---

## 2. 아키텍처 구성도

```
[클라이언트]
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI (app/main.py)                                           │
│  GET /, /ui, /v1/health  │  POST /v1/search, /v1/ask, /v1/upload │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│  Container (app/dependencies.py)                                   │
│  elastic, vector, graph, cache, embedding, reranker, llm_gateway,  │
│  uploads, sources, free_answer, hybrid                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│  HybridSearchService (app/services/hybrid_search_service.py)     │
│  필터 병합 → 캐시 조회/락 → 임베딩 → 4원소 병렬 검색 → RRF 퓨전   │
│  → (선택) LLM 리랭크 → top_k → 캐시 저장                          │
└───┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┘
    │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼
 Redis    Embedding   LLM      Source   Elastic   Milvus
 Cache    Service   Reranker  Service   Client    Client
                                                      │
                                                      ▼
                                                 GraphSearch
                                                 Client(Neo4j)
```

---

## 3. 파일/모듈별 역할 (현재 코드 기준)

### 3.1 진입점·API

| 파일 | 역할 |
|------|------|
| `app/main.py` | FastAPI 앱 생성, lifespan에서 Container 주입, `/` 루트, search/ui 라우터·정적 파일 마운트 |
| `app/api/routes/search.py` | `/v1/search`, `/v1/ask`, `/v1/upload`, `/v1/health`, `/v1/sources`, `/v1/models`, `/v1/cache/*`, `/v1/llm/invoke`, `/v1/free/answer` 등 REST 핸들러 |
| `app/api/routes/ui.py` | UI 라우트 (예: `/ui`) |

### 3.2 의존성·설정

| 파일 | 역할 |
|------|------|
| `app/dependencies.py` | `Container`: ElasticsearchClient, MilvusVectorClient, GraphSearchClient, SearchCache, EmbeddingService, LLMReranker, LLMGateway, UploadDocumentService, SourceService, FreeAnswerService, HybridSearchService 생성 및 `close()` 정리 |
| `app/core/config.py` | pydantic-settings 기반 설정(ES, Milvus, Neo4j, Redis, LLM, Tika, 가중치, 경로 등) |
| `app/core/logging.py` | 로깅 설정 |

### 3.3 검색 파이프라인

| 파일 | 역할 |
|------|------|
| `app/services/hybrid_search_service.py` | 검색 오케스트레이션: 필터 병합, 캐시 키·조회·락, 임베딩, ES/Milvus/Graph/Local 병렬 검색, RRF 퓨전, 선택적 LLM 리랭크, 캐시 저장 |
| `app/services/ranking.py` | **Weighted RRF**: 소스별 순위 리스트와 가중치로 퓨전 점수 계산, `fused_score` 기준 정렬 |

### 3.4 검색 백엔드 클라이언트

| 파일 | 역할 |
|------|------|
| `app/clients/elasticsearch_client.py` | ES 인덱스 존재 시 검색/인덱싱, 쿼리 확장·top-down·function_score 적용, 샘플 모드 시 SampleStore 위임 |
| `app/clients/milvus_client.py` | Milvus 컬렉션/인덱스 확보, 벡터 검색·문서 삽입, 필터 표현식, 샘플 모드 시 SampleStore |
| `app/clients/graph_client.py` | Neo4j fulltext `document_ft` 검색, Document 노드 MERGE·속성 설정, fulltext 인덱스 생성 |

### 3.5 도메인·모델

| 파일 | 역할 |
|------|------|
| `app/domain/construction.py` | TOP_DOWN_LEVELS, ELASTIC_QUERY_FIELDS, TOP_DOWN_FILTER_KEYS, CONSTRUCTION_SYNONYMS, TRADE_PROMPT_TEMPLATES, `merge_top_down_filters`, `expand_construction_query`, `build_context_should_clauses`, `build_function_score_functions`, `infer_trade_from_query`, `build_construction_rerank_prompt` 등 |
| `app/models/schemas.py` | Pydantic 모델: ConstructionTopDownContext, SearchRequest/Response, Candidate, AskRequest/Response, HealthResponse, Cache/Upload/Source/Model/LLM 관련 스키마 |

### 3.6 서비스 계층

| 파일 | 역할 |
|------|------|
| `app/services/cache.py` | Redis 검색 캐시: make_key(SHA256), get_json/set_json, acquire_lock/wait_for_value/release_lock(stampede 방지), invalidate_by_prefix, stats |
| `app/services/embedding_service.py` | OpenAI/Bedrock/Ollama 임베딩, Ollama 다중 URL·모델 탐색, API 불가 시 SHA256 기반 의사 벡터 fallback |
| `app/services/llm_reranker.py` | 건설 도메인 리랭크 프롬프트로 LLM 호출, 응답 JSON의 `ids` 순서로 후보 재정렬 |
| `app/services/llm_gateway.py` | OpenAI 호환/Bedrock/Ollama/Gemini 등 채팅·임베딩 통합 호출 |
| `app/services/source_service.py` | 샘플+업로드 JSON 통합, list_sources, local_search(토큰 매칭+top_down 필터), fallback_context_docs, get_source, read_source_content |
| `app/services/upload_document_service.py` | 파일 수신·저장, 파서 감지, 네이티브 텍스트 추출 → 실패 시 Tika, 문서 객체 구성, ES/Milvus/Neo4j 인덱싱, uploaded_docs.json append |
| `app/services/sample_store.py` | sample_docs + uploaded_docs JSON 로드, keyword_search/graph_search(샘플 모드용) |
| `app/services/free_answer_service.py` | RAG 없이 LLM만으로 답변 (Free API) |

### 3.7 인프라·스키마

| 경로 | 역할 |
|------|------|
| `infra/schemas/elasticsearch_documents_mapping.json` | ES 인덱스 매핑(필요 시) |
| `infra/schemas/neo4j_init.cypher` | Neo4j fulltext/속성 인덱스 정의 |
| `infra/redis/redis.conf` | Redis 서버 설정 |

---

## 4. 핵심 알고리즘 (현재 코드 기준)

### 4.1 Weighted Reciprocal Rank Fusion (RRF)

- **위치**: `app/services/ranking.py` — `weighted_reciprocal_rank_fusion`
- **입력**: `ranked_lists` (소스별 `list[Candidate]`), `weights` (소스별 float), `rrf_k=60`
- **수식**: 문서 `d`의 퓨전 점수  
  `score(d) = Σ_s w_s × 1 / (rrf_k + rank_s(d))`  
  - `rank_s(d)`: 소스 `s`에서 `d`의 1-based 순위
- **동작**: 소스별로 순위 순회하며 점수 누적, 동일 id는 한 번만 선택, `fused_score` 내림차순 정렬 반환
- **설정**: `weight_elastic=0.35`, `weight_vector=0.45`, `weight_graph=0.20`, `weight_local=0.30`

### 4.2 건설 도메인 쿼리 확장 (ES용)

- **위치**: `app/domain/construction.py` — `expand_construction_query`
- **동작**: 쿼리 소문자에 CONSTRUCTION_SYNONYMS 키(slab, pour, rebar, excavation, dewatering, facade, mep, qa, hse)가 포함되면 해당 동의어 리스트를 쿼리 뒤에 공백으로 이어 붙임
- **효과**: Elasticsearch multi_match에서 recall 향상

### 4.3 Elasticsearch 쿼리 구조

- **위치**: `app/clients/elasticsearch_client.py`
- **구성**:
  - **must**: `multi_match` — query=확장 쿼리, fields=ELASTIC_QUERY_FIELDS(title^4, summary^3, content^1.2 등), type=best_fields
  - **should**: `build_context_should_clauses(filters)` — top-down 키별 term 부스트(project_id, building, level 등)
  - **filter**: top_down/일반 필터를 `metadata.*` 또는 필드명 term 쿼리로 적용
  - **function_score**: `build_function_score_functions` — top_down 필터별 가중치, updated_at gauss decay, execution_readiness field_value_factor; score_mode=sum, boost_mode=sum

### 4.4 Milvus 벡터 검색

- **위치**: `app/clients/milvus_client.py`
- **인덱스**: embedding 필드, AUTOINDEX, metric_type=IP(내적)
- **검색**: 쿼리 임베딩 1개, filter=merged_filters를 Milvus 표현식으로 변환, limit=max_candidates_per_source 이내
- **출력**: hit의 distance를 raw_score로 하는 Candidate 리스트

### 4.5 Neo4j Fulltext 검색

- **위치**: `app/clients/graph_client.py`
- **인덱스**: `document_ft` fulltext — Document 노드의 title, content, summary, method_statement, risk_register
- **쿼리**: `CALL db.index.fulltext.queryNodes('document_ft', $q) YIELD node, score` 후 filters 있으면 WHERE로 필터, ORDER BY score DESC LIMIT

### 4.6 로컬 소스 검색 (SourceService)

- **위치**: `app/services/source_service.py`
- **데이터**: `sample_data/sample_docs.json` + `data/uploaded_docs.json` 합친 행 목록
- **토큰화**: `TOKEN_RE`(영문·숫자·한글·_/-)로 쿼리·문서 토큰 집합 생성
- **필터**: selected_source_ids, embedded_only, top_down_filters(키 존재 시 값 일치) 적용
- **스코어**: 쿼리–문서 토큰 교집합 크기 + 제목 토큰 매칭 2배 가산, 선택 소스 보너스
- **fallback_context_docs**: 검색어 매칭 없이 선택/업로드/업로드 시각 등으로 정렬해 문서 보충

### 4.7 LLM 리랭킹

- **위치**: `app/services/llm_reranker.py`, `app/domain/construction.py` — `build_construction_rerank_prompt`
- **입력**: 쿼리, 퓨전 후 상위 50개 Candidate
- **프롬프트**: top-down 우선순위, WBS/CSI/spec, 안전/품질/일정 의도, 최신·실행 준비도; `infer_trade_from_query`로 civil/structure/mep/envelope/safety/general 트레이드별 TRADE_PROMPT_TEMPLATES 문구; 출력 형식 `{"ids": ["id1", "id2", ...]}`
- **동작**: LLM 응답 JSON의 ids 순서로 후보 재정렬 후 top_k 반환; 실패 시 원본 순서 유지

### 4.8 임베딩

- **위치**: `app/services/embedding_service.py`
- **Provider**: openai, bedrock, ollama
- **Ollama**: litellm.aembedding, ollama_base_url(s) 순서 시도, 사용 가능 모델 목록 조회 후 후보 모델로 요청
- **Fallback**: API 불가 시 SHA256 기반 결정론적 의사 벡터(동일 텍스트 → 동일 벡터)
- **차원**: vector_dim=1024, 부족 시 0 패딩 또는 자르기

### 4.9 캐시·동시성

- **위치**: `app/services/cache.py`
- **캐시 키**: 요청 파라미터(query, top_k, filters, selected_source_ids, embedded_only, rerank) JSON을 sort_keys로 직렬화 후 SHA256, 접두어 `search:`
- **저장**: Redis set, TTL=redis_ttl_sec
- **Stampede 방지**: 미스 시 `lock:{key}` NX 락, TTL=redis_lock_ttl_sec; 락 실패 시 wait_for_value(redis_lock_wait_ms 내 폴링)
- **네임스페이스**: REDIS_NAMESPACE 접두어

---

## 5. 문서 파서 전략 (현재 코드 기준)

- **위치**: `app/services/upload_document_service.py`
- **순서**:
  1. 확장자/Content-Type으로 파서 감지(text, pdf, docx, xlsx, pptx, html, hwpx, hwp 등).
  2. **네이티브 추출**: `_extract_text` → 확장자별 `_extract_*` (fitz, docx, pptx, openpyxl, olefile 등).
  3. **Tika 폴백**: 네이티브 결과가 비었고 `tika_url` 설정 시 `_extract_with_tika` 호출 — `/tika/form`, `/tika` PUT, `/rmeta/form` 순서 시도, Accept: text/plain.
  4. 추출 실패 시 본문은 `"Uploaded file: {safe_name}"` 등 대체 텍스트, metadata.parser에 사용한 파서 기록(tika 사용 시 `parser+tika` 또는 `tika`).
- **인덱싱**: 동일 문서를 Elasticsearch, Milvus(임베딩 생성 후), Neo4j, `data/uploaded_docs.json`에 반영.

---

## 6. 검색·RAG 흐름 요약

### POST /v1/search

1. SearchRequest 수신 → merge_top_down_filters  
2. use_cache 시 캐시 키 생성·조회·(미스 시) 락 획득·대기  
3. embed(query) → elastic.search, vector.search, graph.search, source_service.local_search 병렬 실행(asyncio.gather)  
4. weighted_reciprocal_rank_fusion(ranked_lists, weights)  
5. rerank_with_llm이면 reranker.rerank 후 top_k, 아니면 퓨전 결과 상위 top_k  
6. SearchResponse 반환, use_cache 시 캐시 저장·락 해제  

### POST /v1/ask

1. HybridSearchService.search(rerank_with_llm=False)  
2. _filter_hits(selected_source_ids, embedded_only, top_down_context)  
3. 히트 부족 시 local_search·fallback_context_docs로 보강 후 hits[:top_k]  
4. 히트 없음 → 안내 문구 + rag_synthesized=False  
5. 있으면 construction 프롬프트로 llm_gateway.invoke(chat) → 성공 시 답변, 실패 시 _build_extract_answer(검색 요약) + fallback_reason  

---

## 7. 설정 요약 (알고리즘·동작 관련)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| weight_elastic | 0.35 | RRF 키워드(ES) 가중치 |
| weight_vector | 0.45 | RRF 벡터(Milvus) 가중치 |
| weight_graph | 0.20 | RRF 그래프(Neo4j) 가중치 |
| weight_local | 0.30 | RRF 로컬(SourceService) 가중치 |
| max_candidates_per_source | 100 | 소스별 최대 후보 수 |
| vector_dim | 1024 | 임베딩 차원 |
| redis_ttl_sec | 300 | 검색 캐시 TTL |
| redis_lock_ttl_sec | 8 | 캐시 락 TTL |
| redis_lock_wait_ms | 1500 | 락 대기 상한 |
| tika_url | http://tika:9998 | Tika 서버 (문서 파싱 폴백) |
| upstage_dpe_enabled | false | Upstage DPE 폴백 사용 여부 |
| upstage_dpe_min_chars | 80 | 추출 텍스트가 이 값 미만이면 DPE 폴백 시도 |

---

## 8. 데이터 모델 요약

- **Candidate**: id, title, content, source(elastic|vector|graph|local|upload|sample), raw_score, fused_score, metadata  
- **ConstructionTopDownContext**: project_id, building, level, package_code, work_type, task_code, wbs_code 등 (schemas + domain TOP_DOWN_FILTER_KEYS)  
- **SearchRequest**: query, top_k, use_cache, filters, top_down_context, rerank_with_llm, selected_source_ids, embedded_only  
- **AskRequest**: query, top_k, max_tokens, top_down_context, selected_source_ids, embedded_only, provider, model, use_cache  

---

이 설명서는 **현재 코드 기준**으로 작성되었으며, 코드 수정 없이 아키텍처와 알고리즘을 참고하기 위한 문서입니다.

---

## 9. 2026-02 안정성 업데이트

### 9.1 소스 삭제 안정화

- `POST /v1/sources/delete`는 이제 다음 입력을 지원합니다.
  - `source_ids`
  - `delete_all_uploads`
  - `skip_backend_cleanup`
- 전체 선택 삭제 시 대량 ID 전송 없이 `delete_all_uploads=true`로 처리할 수 있습니다.
- 백엔드 정리 삭제는 health 체크 후 살아있는 백엔드만 수행합니다.
- 정리 삭제 호출은 타임아웃/동시성 제한을 적용해 장시간 블로킹을 줄였습니다.

### 9.2 파서 체인 업데이트 (Tika 기본 + DPE 폴백)

- 업로드 파서 순서:
  1. Tika 우선(고급 문서 포맷)
  2. Upstage DPE 폴백(추출 실패 또는 짧은 추출)
  3. 네이티브 파서 폴백
- DPE가 비활성/미설정/불가 상태면 자동 skip되며 업로드는 계속 진행됩니다.

---

## 10. 2026-02 Context Auto-Extraction + Upload Embedding Update

- Upload pipeline now auto-extracts top-down context from filename/content by rule-based parser.
- Optional LLM-based context refinement is supported and controlled by env:
  - `AUTO_CONTEXT_EXTRACT_ENABLED`
  - `AUTO_CONTEXT_EXTRACT_USE_LLM`
  - `AUTO_CONTEXT_EXTRACT_MAX_CHARS`
  - `AUTO_CONTEXT_EXTRACT_MIN_CHARS_FOR_LLM`
  - `AUTO_CONTEXT_EXTRACT_LLM_MODEL`
- Manual upload context fields still override inferred values.
- Upload now always attempts vector embedding/indexing; response includes `embedding_indexed`.
- Upload response also includes `automatic_context_extracted` and final `top_down_context`.
