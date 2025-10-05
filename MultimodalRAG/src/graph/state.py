from typing import Any, Dict, Optional, TypedDict, List
from langchain_core.runnables import Runnable

class GraphState(TypedDict, total=False):
    """State of the graph."""
    question: str
    documents: List[str]
    markdown_docs: List[Any]
    image_docs: List[Any]
    image_contexts: List[str]
    image_paths: List[str]
    image_descriptions: List[Any]
    image_descriptions_text: str
    context: str
    answer: str
    images_context : str
    markdowns_context : str
    markdowns_has_answers : bool
    images_has_answers : bool
    checker_context: str
    answer_image_paths: List[str]
    answer_image_contents: List[str]
    answer_image_justifications: List[str]
    answer_payload: Dict[str, Any]  # Nuevo campo para el payload estructurado