from __future__ import annotations
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "hybrid-search-service"
    app_env: str = "dev"
    app_port: int = 8000
    log_level: str = "INFO"

    elastic_enabled: bool = True
    elastic_url: str = "http://elasticsearch:9200"
    elastic_index: str = "documents"
    elastic_timeout_sec: float = 1.5

    milvus_enabled: bool = True
    milvus_host: str = "milvus-standalone"
    milvus_port: int = 19530
    milvus_collection: str = "document_vectors"
    milvus_timeout_sec: float = 1.5
    vector_dim: int = 1024

    graph_enabled: bool = True
    graph_uri: str = "bolt://neo4j:7687"
    graph_user: str = "neo4j"
    graph_password: str = "neo4j_password"
    graph_database: str = "neo4j"
    graph_timeout_sec: float = 1.5

    redis_enabled: bool = True
    redis_url: str = "redis://redis:6379/0"
    redis_ttl_sec: int = 300
    redis_max_connections: int = 500
    redis_socket_timeout_sec: float = 0.5
    redis_lock_ttl_sec: int = 8
    redis_lock_wait_ms: int = 1500
    redis_namespace: str = "hybrid-search"

    llm_rerank_enabled: bool = False
    llm_provider: str = "ollama"  # openai | openai_compatible | bedrock | ollama | none
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_chat_path: str = "/chat/completions"
    llm_embeddings_path: str = "/embeddings"
    llm_auth_header: str = "Authorization"
    llm_auth_prefix: str = "Bearer "
    llm_embedding_model: str = "text-embedding-3-large"
    llm_rerank_model: str = "gpt-4.1-mini"
    bedrock_region: str = "us-east-1"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v1"
    bedrock_rerank_model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0"
    ollama_base_url: str = "http://localhost:11434"
    ollama_base_urls: str = "http://localhost:11434,http://host.docker.internal:11434"
    ollama_default_model: str = "gpt-oss-120b-cloud"
    llm_timeout_sec: float = 20.0
    ollama_timeout_sec: float = 120.0
    tika_url: str = "http://tika:9998"

    sample_mode: bool = False
    sample_data_path: str = "sample_data/sample_docs.json"
    uploads_data_path: str = "data/uploaded_docs.json"
    uploads_dir: str = "data/uploads"

    weight_elastic: float = Field(default=0.35, ge=0.0)
    weight_vector: float = Field(default=0.45, ge=0.0)
    weight_graph: float = Field(default=0.20, ge=0.0)
    weight_local: float = Field(default=0.30, ge=0.0)
    max_candidates_per_source: int = 100


settings = Settings()





