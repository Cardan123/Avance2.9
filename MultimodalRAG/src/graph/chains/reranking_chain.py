from typing import List, Tuple, Any, Dict
import os
import json
import requests
from loguru import logger
from retrieval import retriever

try:
    from langchain.schema import Document
except Exception:  # fallback shape if LangChain isn't available in this context
    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: Dict[str, Any] | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class RerankingChain:
    """
    Thin wrapper for calling Cohere Rerank API (aka Kohere per request) to re-order documents.

    It expects a partial access to configuration through the `retriever` object to avoid
    reloading config files. We'll look under retriever.vector_search_pipeline.vector_search_config.retrieval_config
    for keys like:
      - cohere_api_key: str
      - reranking: { enabled: bool, model: str, top_n: int }

    Input docs are assumed to be LangChain-like Documents. For now we operate mainly on image_docs,
    but this can be extended to markdown_docs easily.
    """

    DEFAULT_MODEL = "rerank-english-v3.0"  # reasonable default; adjust if multilingual needed
    def __init__(self, retriever: retriever.Retriever):
        self.model: str = retriever.vector_search_pipeline.vector_search_config.reranking_model or self.DEFAULT_MODEL
        self.top_n: int = int(retriever.vector_search_pipeline.vector_search_config.reranking_top_n or 10)
        # API key location; allow env override
        self.api_key: str | None = os.getenv("COHERE_API_KEY") or (self.retrieval_config or {}).get("cohere_api_key")

    def _cohere_rerank(self, query: str, documents: List[Document]) -> List[int]:
        """
        Calls Cohere Rerank API and returns the ranking as indices into `documents` (sorted best->worst).
        """
        if not self.api_key:
            logger.warning("[RerankingChain] COHERE_API_KEY missing. Skipping rerank.")
            return list(range(len(documents)))

        url = "https://api.cohere.com/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        items = [
            {"text": getattr(doc, "page_content", str(doc)) or ""}
            for doc in documents
        ]

        payload = {
            "model": self.model,
            "query": query or "",
            "documents": items,
            "top_n": min(self.top_n, len(items)) if self.top_n else len(items),
        }

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if resp.status_code != 200:
                logger.warning("[RerankingChain] Cohere rerank bad status {}: {}", resp.status_code, resp.text[:200])
                return list(range(len(documents)))
            data = resp.json()
            # Expected data structure includes 'results' with 'index' and 'relevance_score'
            results = data.get("results") or []
            # Sort by score desc if needed (Cohere already returns sorted, but ensure)
            results_sorted = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
            ranked_indices = [int(r.get("index", i)) for i, r in enumerate(results_sorted) if isinstance(r, dict)]
            # Ensure valid bounds
            ranked_indices = [i for i in ranked_indices if 0 <= i < len(documents)]
            if not ranked_indices:
                return list(range(len(documents)))
            return ranked_indices
        except Exception as e:
            logger.error("[RerankingChain] Error calling Cohere rerank: {}", e)
            return list(range(len(documents)))

    def rerank_documents(self, question: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        if not self.enabled:
            logger.info("[RerankingChain] Reranking disabled by config. Returning original order.")
            return docs
        ranked = self._cohere_rerank(question, docs)
        # reorder by ranked list; if ranked shorter, append remaining
        used = set(ranked)
        ordered = [docs[i] for i in ranked]
        for i in range(len(docs)):
            if i not in used:
                ordered.append(docs[i])
        return ordered
