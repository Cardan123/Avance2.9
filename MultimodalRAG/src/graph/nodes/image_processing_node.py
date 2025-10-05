from loguru import logger
import os
from pathlib import Path
from utils.config_file_manager import ConfigFileManager
from utils.path_normalizer import normalize_control_path
from retrieval import retriever  # exported via retrieval/__init__.py
from graph.consts import IMAGE_VERBALIZATION, END,IMAGE_CONTEXT

class ImageProcessingNode:
    @staticmethod
    def process(state, retriever: retriever.Retriever = None):
        """Procesa imágenes y genera contexto preservando la subruta después de 'data'."""
        #config_path = ConfigFileManager.default_yaml_path()
        #config = ConfigFileManager.load_yaml_config(config_path)
        config = retriever.vector_search_pipeline.vector_search_config.retrieval_config
        base_images_path = config.get("base_images_path", "")
        image_contexts = []
        image_docs = state.get("image_docs", [])
        valid_image_paths = []  # NUEVO

        for image_doc in image_docs:
            # Manejo flexible de distintos tipos de "Document"
            metadata = ImageProcessingNode.extract_metadata(image_doc)

            if not isinstance(metadata, dict):
                # A veces metadata puede venir como None u otro tipo
                metadata = {}
                logger.warning("[ImageProcessing Node] Metadata is not a dict, defaulting to empty dict.")
                continue

            source_file = ImageProcessingNode.extract_source_file(metadata)

            if not source_file:
                logger.warning("[ImageProcessing Node] No source file detected in image document.")
                continue

            relative_subpath = None
            try:
                p = Path(source_file)
                parts = p.parts
                for i, part in enumerate(parts):
                    if part.lower() == "data":
                        after_parts = parts[i + 1:]
                        if after_parts:
                            relative_subpath = Path(*after_parts)
                        break
            except Exception as e:
                logger.warning(f"[ImageProcessing Node] Error parsing path: {e}")

            if relative_subpath is None:
                relative_subpath = Path(os.path.basename(source_file))

            if base_images_path:
                final_path = os.path.normpath(str(Path(base_images_path) / relative_subpath))
            else:
                final_path = os.path.normpath(source_file)

            final_path = normalize_control_path(final_path)
            if not os.path.isfile(final_path):
                logger.warning(f"[ImageProcessing Node] Image file does not exist: {final_path}")
                continue

            valid_image_paths.append(final_path)  # NUEVO
            image_contexts.append(ImageProcessingNode.extract_content(image_doc))

        state["image_contexts"] = image_contexts
        state["image_paths"] = valid_image_paths  # NUEVO
        logger.info(f"[ImageProcessing Node] Relevant Image paths extracted:\n {valid_image_paths}")
        logger.info(f"[ImageProcessing Node]: Generated context for {len(image_contexts)} images.")
        return state

    @staticmethod
    def extract_metadata(image_doc):
        """Extrae un dict de metadata desde distintos tipos de objetos/documentos."""
        if isinstance(image_doc, dict):
            metadata = image_doc.get("metadata", {})
        else:
            if hasattr(image_doc, "metadata"):
                metadata = getattr(image_doc, "metadata") or {}
            elif hasattr(image_doc, "meta"):
                metadata = getattr(image_doc, "meta") or {}
            elif hasattr(image_doc, "__dict__"):
                metadata = getattr(image_doc, "__dict__", {}).get("metadata", {}) or {}
            else:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata
    
    @staticmethod
    def extract_source_file(md: dict):
            if not isinstance(md, dict):
                return None
            candidate_keys = ("source_file", "source", "file_path", "path")
            # Nivel directo
            for k in candidate_keys:
                v = md.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            # Posibles contenedores anidados
            nested_keys = ("paths", "pats", "files", "images", "items")
            for nk in nested_keys:
                nv = md.get(nk)
                if isinstance(nv, dict):
                    # Buscar directo dentro del dict
                    for k in candidate_keys:
                        v = nv.get(k)
                        if isinstance(v, str) and v.strip():
                            return v
                elif isinstance(nv, list):
                    for item in nv:
                        if isinstance(item, dict):
                            for k in candidate_keys:
                                v = item.get(k)
                                if isinstance(v, str) and v.strip():
                                    return v
            return None
    
    @staticmethod
    def extract_content(doc):
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
    