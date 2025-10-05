from loguru import logger


class MarkdownContextNode:
    """Build a list of markdown page contents from the retrieved documents.

    Expects in state:
        - documents: iterable of Document-like objects (with .page_content & .metadata) or dicts.
    Produces in state:
        - markdown_contexts: list[str] of page_content for markdown documents
        - markdowns_context: single concatenated string (joined with newlines)
    """

    @staticmethod
    def process(state):
        documents = state.get("documents", []) or []

        def _is_markdown(doc):
            try:
                # LangChain Document style
                meta = getattr(doc, "metadata", None)
                if isinstance(meta, dict):
                    return meta.get("doc_type") == "markdown"
                # Dict style
                if isinstance(doc, dict):
                    return doc.get("metadata", {}).get("doc_type") == "markdown" or doc.get("doc_type") == "markdown"
            except Exception:
                return False
            return False

        def _extract_content(doc):
            if hasattr(doc, "page_content"):
                return getattr(doc, "page_content", "")
            if hasattr(doc, "content"):
                return getattr(doc, "content", "")
            if isinstance(doc, dict):
                # Possible shapes
                if "page_content" in doc:
                    return doc.get("page_content") or ""
                return doc.get("content") or doc.get("text") or ""
            # Fallback to string representation
            return str(doc)

        # Filter only markdown documents (if any). If none flagged, use all.
        markdown_docs = [d for d in documents if _is_markdown(d)] or documents

        markdown_contexts = []
        for doc in markdown_docs:
            content = _extract_content(doc)
            if content:
                markdown_contexts.append(content)

        state["markdown_contexts"] = markdown_contexts
        state["markdowns_context"] = "\n".join(markdown_contexts)

        logger.info(
            f"[Markdown Context Node] Generated markdown_contexts list with {len(markdown_contexts)} items."
        )
        return state