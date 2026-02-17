from __future__ import annotations
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ConstructionTopDownContext(BaseModel):
    portfolio_id: Optional[str] = None
    program_id: Optional[str] = None
    project_id: Optional[str] = None
    site_code: Optional[str] = None
    asset_type: Optional[str] = None
    zone: Optional[str] = None
    building: Optional[str] = None
    level: Optional[str] = None
    package_code: Optional[str] = None
    work_type: Optional[str] = None
    discipline: Optional[str] = None
    activity: Optional[str] = None
    task_code: Optional[str] = None
    wbs_code: Optional[str] = None
    csi_division: Optional[str] = None
    spec_section: Optional[str] = None
    contractor: Optional[str] = None
    safety_risk_class: Optional[str] = None
    quality_check_type: Optional[str] = None
    schedule_window: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=20, ge=1, le=200)
    use_cache: bool = True
    filters: Optional[Dict[str, Union[str, int, float, bool]]] = None
    top_down_context: Optional[ConstructionTopDownContext] = None
    rerank_with_llm: bool = False
    selected_source_ids: Optional[List[str]] = None
    embedded_only: bool = False


class Candidate(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    source: str
    raw_score: float = 0.0
    fused_score: float = 0.0
    metadata: Dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    took_ms: int
    hits: List[Candidate]
    cache_hit: bool = False


class HealthResponse(BaseModel):
    status: str
    elastic: str
    milvus: str
    graph: str
    redis: str


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMInvokeRequest(BaseModel):
    task: str = Field(default="chat")  # chat | embed
    provider: Optional[str] = None  # openai | openai_compatible | bedrock | ollama
    model: Optional[str] = None
    text: Optional[str] = None
    messages: Optional[List[LLMMessage]] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    region: Optional[str] = None


class LLMInvokeResponse(BaseModel):
    provider: str
    model: str
    task: str
    output_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    raw: Dict = Field(default_factory=dict)


class CacheStatsResponse(BaseModel):
    status: str
    key_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    set_count: int = 0
    delete_count: int = 0
    lock_contention_count: int = 0


class CacheInvalidateRequest(BaseModel):
    prefix: str = Field(default="search:")


class UploadResponse(BaseModel):
    file_name: str
    doc_id: str
    indexed: bool
    chars_indexed: int
    parser: str = "fallback"
    indexed_backends: List[str] = Field(default_factory=list)
    extracted: bool = True
    warning: Optional[str] = None


class FreeAnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1024)


class FreeAnswerResponse(BaseModel):
    provider: str
    query: str
    answer: str
    source_url: Optional[str] = None


class SourceItem(BaseModel):
    id: str
    title: str
    source_type: str
    embedded: bool = False
    selected: bool = False
    metadata: Dict = Field(default_factory=dict)


class SourceListResponse(BaseModel):
    items: List[SourceItem]
    count: int


class ModelItem(BaseModel):
    provider: str
    model: str
    available: bool = True


class ModelListResponse(BaseModel):
    items: List[ModelItem]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=8, ge=1, le=50)
    max_tokens: Optional[int] = Field(default=64, ge=1, le=8192)
    top_down_context: Optional[ConstructionTopDownContext] = None
    selected_source_ids: Optional[List[str]] = None
    embedded_only: bool = False
    provider: str = "ollama"
    model: Optional[str] = None
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    provider: str
    model: str
    hits: List[Candidate]




