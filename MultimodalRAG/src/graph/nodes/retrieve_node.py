from typing import Any, Dict, Optional
from loguru import logger
from graph.state import GraphState

class RetrieveNode:
    """
    Node to retrieve documents based on a question.
    """
    @staticmethod
    def process(state: GraphState, retriever) -> Dict[str, Any]:
        """Retrieve documents based on the question in the state."""

        logger.info("[Retrieve Node] Starting document retrieval...")
        question = state["question"]
        logger.info(f"[Retrieve Node] Retrieving documents for question: {question}")
        ctx, docs = retriever.vector_search_pipeline.execute_vector_search(question)
        logger.info(f"[Retrieve Node] Retrieved {len(docs)} documents.")

        logger.info("[Retrieve Node] Adding documents into graph state.")
        state["documents"] = docs
        
        return state