# Plan -> MVP -> Production

## Phase 0: Scope and SLO
- Data scale target: 100TB raw documents, 30TB embeddings.
- Retrieval target: hybrid retrieval per query (keyword + vector + graph).
- Service SLO target:
  - P95 < 1.5s on cache miss.
  - P95 < 150ms on cache hit.
  - Partial-result response on engine failure (degraded mode).

## Phase 1: MVP (Implemented)
1. FastAPI search API with one hybrid endpoint.
2. Parallel fan-out to Elasticsearch, Milvus, Neo4j.
3. Weighted RRF for result fusion.
4. Redis result cache.
5. Optional LLM reranking.
6. Nginx load balancing with 2 API replicas.
7. Construction top-down context model and filter merge.

## Phase 2: Data Pipeline
1. Raw document pipeline:
   - Ingest -> normalize -> chunk -> enrich metadata -> index.
2. Embedding pipeline:
   - Chunk -> embed -> Milvus upsert with versioning.
3. Graph pipeline:
   - Entity extraction -> relation build -> Neo4j upsert.
4. Incremental updates:
   - CDC + idempotent upsert and delete propagation.
5. Index lifecycle:
   - Blue/green index alias migration for safe rollout.

## Phase 3: Production Hardening
1. Reliability:
   - Retry budget, timeout budget, circuit breaker, bulkhead.
2. Scale:
   - Kubernetes HPA + PDB for API.
   - ES hot/warm tiers and ILM.
   - Milvus partition and compaction strategy.
   - Neo4j cluster topology.
3. Observability:
   - OpenTelemetry tracing.
   - Metrics: QPS, P95, cache hit ratio, per-engine latency/error.
   - SLO alerting.
4. Security:
   - Secret manager, private network, TLS, IAM least privilege.

## Phase 4: Relevance Quality
1. Domain query rewrite (construction synonyms and aliases).
2. LTR or cross-encoder rerank for high-value queries.
3. Human feedback loop and judgment set governance.
4. Offline and online relevance tracking.

## Operational Checklist
- [ ] Fixed schema contract for metadata fields.
- [ ] Shard and replica sizing based on load test.
- [ ] Backup and restore drill completed.
- [ ] Error budget policy defined.
- [ ] Cost controls for storage, embedding, and inference.
