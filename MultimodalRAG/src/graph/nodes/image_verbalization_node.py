from loguru import logger
from retrieval.image_description_chain import build_image_description_chain
from utils.model_manager import GeminiLLMWrapper  # exported via retrieval/__init__.py
from retrieval import retriever  # exported via retrieval/__init__.py

class ImageVerbalizationNode:
    @staticmethod
    def process(state = None, retriever: retriever.Retriever = None):
        """Toma state["image_paths"], genera descripciones y las agrega al estado.

        Añade:
            state["image_descriptions"]: lista de dicts (uno por imagen)
            state["image_descriptions_text"]: concatenación de descripciones crudas (para inyectar en RAG si se requiere)
        """
        image_paths = state.get("image_paths", [])
        if not image_paths:
            logger.warning("[ImageVerbalizationNode] No image_paths in state.")
            state["image_descriptions"] = []
            state["image_descriptions_text"] = ""
            return state

        logger.info(f"[ImageVerbalizationNode] Describing {len(image_paths)} images with Gemini...")

        # Reusar wrapper (idealmente pasar llm desde fuera para no recrear)
        #llm = GeminiLLMWrapper().llm
        llm = retriever.gemini_llm_wrapper.llm
        chain = build_image_description_chain(llm=llm)
        results = chain.run(image_paths=image_paths, extra_instruction="Enfoca solo información útil.")

        # Build aggregated text (only raw parts)
        aggregated_text = "\n".join(
            r.get("raw_description", "") for r in results if not r.get("error")
        )

        state["image_descriptions"] = results
        state["image_descriptions_text"] = aggregated_text
        logger.info("[ImageVerbalizationNode] Image descriptions added to state.")
        return state
