from loguru import logger
import os

class ContextingNode:
    @staticmethod
    def process(state):
        """Construye el contexto final en `state["context"]` para el nodo Generate.

        Fuentes utilizadas (todas opcionales):
        - state["markdowns_context"]: Texto consolidado de documentos markdown.
        - state["answer_image_paths"], state["answer_image_contents"], state["answer_image_justifications"]: Evidencia visual relevante seleccionada por el checker.
        - (fallback) state["images_context"] si no hay imágenes con respuestas pero sí descripciones generales.

        Reglas:
        1. Siempre incluir sección textual si existe (`markdowns_context`).
        2. Si hay imágenes con respuestas (answer_image_paths no vacío) se agrega sección "Evidencia Visual Relevante" formateada.
        3. Si NO hay imágenes con respuestas pero existe `images_context`, añadir como "Contexto de Imágenes" (legacy).
        4. Formato markdown para facilitar extracción posterior por el LLM.
        """

        # --- Recuperar componentes ---
        checker_ctx = state.get("checker_context") or ""
        md_ctx = (state.get("markdowns_context") or "").strip()
        answer_paths = state.get("answer_image_paths") or []
        answer_contents = state.get("answer_image_contents") or []
        answer_justifs = state.get("answer_image_justifications") or []
        legacy_images_ctx = (state.get("images_context") or "").strip()

        sections = []

        # --- Sección textual ---
        if md_ctx:
            sections.append("### Contexto Textual\n" + md_ctx)
        else:
            logger.info("[ContextingNode] No markdown textual context found.")

        

        # --- Evidencia visual prioritaria ---
        use_answer_parts = True

        if not checker_ctx:
            use_answer_parts = True

        if not use_answer_parts:
            if checker_ctx:
                sections.append("### Contexto del Checker\n" + checker_ctx)

        if use_answer_parts and answer_paths:
            visual_lines = ["### Evidencia Visual Relevante"]
            for idx, (p, content, justif) in enumerate(zip(answer_paths, answer_contents, answer_justifs), start=1):
                fname = os.path.basename(p)
                block_parts = [f"#### Imagen {idx}: {fname}"]
                block_parts.append(f"Ruta: {p}")
                if content:
                    block_parts.append("Respuesta extraída: " + content.strip())
                if justif:
                    block_parts.append("Justificación: " + justif.strip())
                visual_lines.append("\n".join(block_parts))
            sections.append("\n\n".join(visual_lines))
            logger.info("[ContextingNode] Added visual evidence section with {} images.", len(answer_paths))
        elif use_answer_parts and legacy_images_ctx:
            # Solo si no hubo imágenes con respuesta pero existe contexto descriptivo
            sections.append("### Contexto de Imágenes (Descriptivo)\n" + legacy_images_ctx)
            logger.info("[ContextingNode] Added legacy images descriptive context (no answer images).")
        else:
            logger.info("[ContextingNode] No visual evidence or descriptive image context to add.")

        if not sections:
            logger.warning("[ContextingNode] No context sections available; `context` quedará vacío.")

        consolidated = "\n\n".join([s for s in sections if s.strip()])

        # Guardar en el estado bajo la clave requerida por GenerateNode
        state["context"] = consolidated
        logger.info("[ContextingNode] Context final construido (len={} chars, secciones={}).", len(consolidated), len(sections))
        return state
