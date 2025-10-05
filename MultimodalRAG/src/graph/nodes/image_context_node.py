from loguru import logger
import json
import re
import os
from typing import List, Dict, Any
from pathlib import Path
from graph.consts import IMAGE_CHECKER, CONTEXTING, END

class ImageContextNode:
    @staticmethod
    def process(state):
        """Genera contexto enriquecido para imágenes, integrando descripciones y metadatos.
            Construye `state["images_context"]` con formato:
            [IMAGE] filename: <nombre archivo>
            CONTENT: <contenido enriquecido>
            ---
            """
        images_context = ImageContextNode.build_checker_context(state)
        state["images_context"] = images_context

        logger.info("[Image Context Node] Context generated (includes images: {}).",
                    "yes" if images_context else "no")

        return state
    
    @staticmethod
    def build_checker_context(state: Dict[str, Any]) -> str:
        image_paths: List[str] = state.get("image_paths") or []
        image_contexts: List[str] = state.get("image_contexts") or []
        descriptions_raw = state.get("image_descriptions_text") or ""

        parsed_desc: Dict[str, Dict[str, Any]] = {}
        if descriptions_raw:
            fence_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
            snippets: List[str] = []
            matches = fence_pattern.findall(descriptions_raw)
            if matches:
                snippets.extend(matches)
            else:
                snippets.append(descriptions_raw)
            for snippet in snippets:
                snippet_clean = snippet.strip().strip('`')
                try:
                    data = json.loads(snippet_clean)
                    if isinstance(data, dict):
                        fname = data.get("archivo") or data.get("file")
                        if fname:
                            parsed_desc[os.path.basename(fname)] = data
                except Exception:
                    continue

        lines: List[str] = []
        for idx, path in enumerate(image_paths):
            name = os.path.basename(path)
            base_content = image_contexts[idx] if idx < len(image_contexts) else ""
            enrich_parts: List[str] = []
            desc = parsed_desc.get(name)
            if desc:
                if desc.get("resumen"):
                    enrich_parts.append(f"Resumen: {desc.get('resumen')}")
                if desc.get("objetos"):
                    objs = desc.get("objetos")
                    if isinstance(objs, list):
                        objs = ", ".join(objs)
                    enrich_parts.append(f"Objetos: {objs}")
                if desc.get("texto"):
                    txts = desc.get("texto")
                    if isinstance(txts, list):
                        txts = ", ".join(txts[:8])
                    enrich_parts.append(f"Texto: {txts}")
            full_content = base_content
            if enrich_parts:
                full_content = (full_content + "\n" + " | ".join(enrich_parts)).strip()
            if len(full_content) > 2000:
                full_content = full_content[:2000] + "..."
            lines.append(f"[IMAGE] filename: {name}\nCONTENT: {full_content}\n---")

        if not lines and state.get("images_context"):
            lines.append(state.get("images_context"))

        checker_context = "\n".join(lines)
        return checker_context
    
    @staticmethod
    def route_from_image_context(state: Dict[str, Any], retriever=None) -> str:
        """
        Decide the next step after reranking. We want to go to IMAGE_PROCESSING when image docs exist.
        If reranking is disabled in config, this router should still be called but simply defer to the
        regular flow based on content availability.

        It receives retriever partially so we can read the configuration loaded already.
        """
        try:
            checker_enabled = bool(retriever.vector_search_pipeline.vector_search_config.checker_enabled)

            if checker_enabled:
                return IMAGE_CHECKER
            else:
                return CONTEXTING
        except Exception:
            return END