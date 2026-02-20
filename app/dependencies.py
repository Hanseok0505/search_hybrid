from __future__ import annotations
from app.clients.elasticsearch_client import ElasticsearchClient
from app.clients.graph_client import GraphSearchClient
from app.clients.milvus_client import MilvusVectorClient
from app.services.cache import SearchCache
from app.services.embedding_service import EmbeddingService
from app.services.free_answer_service import FreeAnswerService
from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_gateway import LLMGateway
from app.services.llm_reranker import LLMReranker
from app.services.source_service import SourceService
from app.services.upload_document_service import UploadDocumentService


class Container:
    def __init__(self) -> None:
        self.elastic = ElasticsearchClient()
        self.vector = MilvusVectorClient()
        self.graph = GraphSearchClient()
        self.cache = SearchCache()
        self.embedding = EmbeddingService()
        self.reranker = LLMReranker()
        self.llm_gateway = LLMGateway()
        self.uploads = UploadDocumentService(
            elastic=self.elastic,
            vector=self.vector,
            graph=self.graph,
            embedding=self.embedding,
            llm_gateway=self.llm_gateway,
        )
        self.sources = SourceService()
        self.free_answer = FreeAnswerService()
        self.hybrid = HybridSearchService(
            elastic=self.elastic,
            vector=self.vector,
            graph=self.graph,
            cache=self.cache,
            embedding=self.embedding,
            reranker=self.reranker,
            source_service=self.sources,
        )

    async def close(self) -> None:
        await self.elastic.close()
        await self.graph.close()
        await self.cache.close()
        await self.embedding.close()
        await self.reranker.close()
        await self.llm_gateway.close()
        await self.free_answer.close()




