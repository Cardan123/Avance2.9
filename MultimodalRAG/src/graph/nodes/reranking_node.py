from typing import Any, Dict, List
from loguru import logger
from zmq import has

from graph.consts import IMAGE_PROCESSING, CONTEXTING, END, RERANKING
from graph.chains.reranking_chain import RerankingChain


class RerankingNode:
    @staticmethod
    def process(state: Dict[str, Any], retriever=None) -> Dict[str, Any]:
        """
        Rerank retrieved documents. For now, operate on image_docs only. In the future, also on markdown_docs.

        Expects state keys:
          - question: str
          - documents: List[Document-like]
        Produces/updates:
          - documents (reordered) and the split lists accordingly
          - image_docs (reordered subset)
        """
        logger.info("[Reranking Node] Starting reranking process...")
        try:
            question: str = state.get("question") or ""
            documents: List[Any] = state.get("image_docs", []) or []

            if not documents:
                logger.info("[Reranking Node] No documents present; skipping.")
                return state

            chain = RerankingChain(retriever)
            # Separate docs by type
            def _is_image(doc) -> bool:
                try:
                    md = getattr(doc, "metadata", None)
                    if isinstance(md, dict):
                        return (md.get("doc_type") or md.get("type")) == "image"
                    if isinstance(doc, dict):
                        return doc.get("metadata", {}).get("doc_type") == "image" or doc.get("doc_type") == "image"
                except Exception:
                    return False
                return False

            image_docs = [d for d in documents if _is_image(d)]
            markdown_docs = [d for d in documents if not _is_image(d)]

            if not image_docs:
                logger.info("[Reranking Node] No image docs found; leaving order unchanged.")
                return state

            reranked_images = chain.rerank_documents(question, image_docs)

            # Rebuild documents: keep non-image order as-is, prepend reranked images first or interleave?
            # For safety, place reranked images first to increase their downstream priority.
            new_documents = reranked_images + markdown_docs
            #state["documents"] = new_documents
            state["image_docs"] = reranked_images

            logger.info("[Reranking Node] Reranked {} image docs.", len(reranked_images))
            return state
        except Exception as e:
            logger.error("[Reranking Node] Error during reranking: {}", e)
            return state

    @staticmethod
    def route_from_reranking(state: Dict[str, Any], retriever=None) -> str:
        """
        Decide the next step after reranking. We want to go to IMAGE_PROCESSING when image docs exist.
        If reranking is disabled in config, this router should still be called but simply defer to the
        regular flow based on content availability.

        It receives retriever partially so we can read the configuration loaded already.
        """
        try:
            reranking_enabled = bool(retriever.vector_search_pipeline.vector_search_config.retrieval_config.get("reranking", {}).get("enabled", False))
            has_images = bool(state.get("image_docs"))
            if reranking_enabled:
                if has_images:
                    return RERANKING
                # If there are no images but there are markdowns, go to CONTEXTING; else END
                md = state.get("markdown_docs") or []
                if md:
                    return CONTEXTING
                return END
            if not reranking_enabled and has_images:
                return IMAGE_PROCESSING
            if has_images:
                return IMAGE_PROCESSING
            # If there are no images but there are markdowns, go to CONTEXTING; else END
            md = state.get("markdown_docs") or []
            if md:
                return CONTEXTING
            return END
        except Exception:
            return END
