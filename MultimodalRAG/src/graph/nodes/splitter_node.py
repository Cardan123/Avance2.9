from loguru import logger
from graph.consts import IMAGE_PROCESSING, CONTEXTING, END, RERANKING
from typing import Any, Dict


class SplitterNode:
    @staticmethod
    def process(state):
        """Divide documentos según su doc_type."""
        markdown_docs = SplitterNode.get_markdown_docs(state)
        image_docs = SplitterNode.get_image_docs(state)

        #Add a document dumb for testing in image_docs if empty
        if not image_docs and False: # Change to True to enable dummy image doc
            image_docs.append({
                "metadata": {"doc_type": "image", 
                             "source_file": "D:\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Github\MultimodalRAG\data\Edificio Tello\VIEW SCREENSHOTS\01\01_10-01-28-01.png"},
                "page_content": "Dummy image content"
            })

        logger.info(f"[Splitter Node]: Detected {len(markdown_docs)} markdown docs, {len(image_docs)} image docs found.")

        state["markdown_docs"] = markdown_docs
        state["image_docs"] = image_docs
        return state

    @staticmethod
    def has_images(state):
        """Condición: Verifica si hay documentos de tipo imagen."""
        return bool(state.get("image_docs"))

    @staticmethod
    def has_markdown(state):
        """Condición: Verifica si hay documentos de tipo markdown."""
        return bool(state.get("markdown_docs"))
    
    @staticmethod
    def route_from_splitter(state):
        # Si hay imágenes, procesar imágenes primero
        if SplitterNode.has_images(state):
            return IMAGE_PROCESSING
        # Si solo hay markdown, ir al CONTEXTING (ya no directo a CONTEXTING para evitar ramas paralelas)
        if SplitterNode.has_markdown(state):
            return CONTEXTING
        # Si no hay nada, terminar
        return END
    
    @staticmethod
    def route_from_splitter_reranking(state: Dict[str, Any], retriever=None) -> str:
        """
        Decide the next step after reranking. We want to go to IMAGE_PROCESSING when image docs exist.
        If reranking is disabled in config, this router should still be called but simply defer to the
        regular flow based on content availability.

        It receives retriever partially so we can read the configuration loaded already.
        """
        try:
            reranking_enabled = bool(retriever.vector_search_pipeline.vector_search_config.reranking_enabled)
            has_images = SplitterNode.has_images(state)
            has_markdowns = SplitterNode.has_markdown(state)
            if reranking_enabled:
                if has_images:
                    return RERANKING
                # If there are no images but there are markdowns, go to CONTEXTING; else END
                if has_markdowns:
                    return CONTEXTING
                return END
            if not reranking_enabled and has_images:
                return IMAGE_PROCESSING
            if has_images:
                return IMAGE_PROCESSING
            # If there are no images but there are markdowns, go to CONTEXTING; else END
            if has_markdowns:
                return CONTEXTING
            return END
        except Exception:
            return END
    
    @staticmethod
    def get_markdown_docs(state):
        """Obtiene documentos de tipo markdown."""
        return [doc for doc in state["documents"] if doc.metadata.get("doc_type") == "markdown"]

    @staticmethod
    def get_image_docs(state):
        """Obtiene documentos de tipo imagen."""
        return [doc for doc in state["documents"] if doc.metadata.get("doc_type") == "image"]