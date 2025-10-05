from loguru import logger
from graph.state import GraphState
from typing import Any, Dict
from langchain_core.runnables import Runnable
from graph.chains.generate_chain import GenerateChain
from retrieval import retriever  # exported via retrieval/__init__.py

class GenerateNode:
    @staticmethod
    def generate(state : GraphState, retriever: retriever.Retriever) -> Dict[str, Any]:
        """Genera el prompt final combinando el prompt base con los contextos de las imágenes."""
        logger.info("[Generate Node]: Generating final answer.")
        #chain = GenerateChain(llm=retriever.gemini_llm_wrapper.llm,
        #                     prompt=retriever.prompt_template_wrapper.create_generic_prompt_template())
        chain = retriever.llm_chain_creator.llm_chain

        # --- Diagnóstico adicional: verificar contexto antes de invocar el LLM ---
        context_value = state.get("context", "")
        if not context_value:
            logger.warning("[Generate Node] state['context'] está vacío. The model may not respond with the answer.")
            # Fallback: si existen documentos en state y no se construyó el contexto, concatenarlos rápidamente
            docs = state.get("documents") or []
            if docs:
                try:
                    quick_ctx = " \n".join(getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", ""))
                    if quick_ctx:
                        context_value = quick_ctx[:8000]  # recorta para no exceder token context si es muy grande
                        logger.warning("[Generate Node] Using fallback for document concatenation ({} chars).", len(context_value))
                except Exception as e:
                    logger.error("[Generate Node] Error creando fallback de contexto: {}", e)

        logger.debug("[Generate Node] Context length={} chars. Preview: {}", len(context_value), context_value[:20].replace("\n", " "))

        payload = {
            "question": state.get("question", ""),
            "document": context_value
            # Pre-procesa el contexto: antepone un bloque 'Relevant Context' con líneas que contengan palabras clave de la pregunta
            #"document": GenerateNode._prioritize_relevant_snippets(state.get("question", ""), context_value)
        }
        logger.debug("[Generate Node] Sending payload to the LLM payload keys={}.", list(payload.keys()))

        result = chain.invoke(payload)

        if isinstance(result, dict):
            answer = result.get("answer") or result.get("text")
        else:
            answer = result

        if answer is None:
            logger.warning("[Generate Node] Output didn't contain 'answer' or 'text' key. Raw result: {}", result)
            answer = ""

        state["answer"] = answer
        state["answer_payload"] = GenerateNode.get_payload_answeranswer(state)  # Actualiza state con payload estructurado
        logger.info("[Generate Node] Generated answer (len={} chars).", len(answer))
        return state

    @staticmethod
    def get_payload_answeranswer(state: GraphState) -> Dict[str, Any]:
        """Parser utilitario para construir un diccionario estructurado con la respuesta y sus referencias.

        Construye un objeto estilo payload para consumo externo (API / UI):
            {
              "answer": <str>,
              "AnswerReferences": [<image_path_1>, <image_path_2>, ...]
            }

        Referencias se toman de `state['answer_image_paths']` (imágenes evaluadas como relevantes por el checker).
        Si esa lista está vacía, hace fallback opcional a `state['image_paths']` solo si se desea mostrar algo (pero
        aquí preferimos devolver lista vacía para no introducir ruido no validado).
        """
        answer_text = state.get("answer", "") or ""
        refs = state.get("answer_image_paths") or []
        # Asegurar unicidad y mantener orden de aparición
        seen = set()
        ordered_unique_refs = []
        for p in refs:
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                ordered_unique_refs.append(p)
        payload = {
            "answer": answer_text.strip(),
            "answer_references": ordered_unique_refs,
        }
        logger.debug("[Generate Node] parse_answer produced payload with {} references.", len(ordered_unique_refs))
        return payload

    @staticmethod
    def _prioritize_relevant_snippets(question: str, full_context: str, max_lines: int = 12) -> str:
        """Extrae líneas relevantes al question (heurística simple) y las antepone como 'Relevant Context'.

        - Busca tokens significativos (>=3 chars) de la pregunta.
        - Conserva las primeras coincidencias distintas.
        - Evita duplicar si ya es muy corto o si no hay coincidencias.
        """
        try:
            if not question or not full_context:
                return full_context

            import re
            # Normalizar y tokenizar pregunta
            tokens = [t.lower() for t in re.split(r"[^\wáéíóúüñ']+", question) if len(t) >= 3]
            tokens = list(dict.fromkeys(tokens))  # únicos preservando orden
            if not tokens:
                return full_context

            lines = [ln for ln in full_context.splitlines() if ln.strip()]
            scored = []
            for ln in lines:
                low = ln.lower()
                score = sum(1 for tk in tokens if tk in low)
                if score > 0:
                    scored.append((score, ln))

            if not scored:
                return full_context

            # Orden por score desc y longitud moderada (líneas cortas primero a igualdad)
            scored.sort(key=lambda x: (-x[0], len(x[1])))
            selected = []
            seen = set()
            for score, ln in scored:
                norm = ln.strip()
                if norm in seen:
                    continue
                seen.add(norm)
                selected.append(ln)
                if len(selected) >= max_lines:
                    break

            if not selected:
                return full_context

            relevant_block = "Relevant Context (auto-extracted):\n" + "\n".join(selected) + "\n\n" + full_context
            # Limitar tamaño para no desbordar contexto del modelo (heurístico)
            if len(relevant_block) > 24000:
                return relevant_block[:24000]
            return relevant_block
        except Exception as e:
            logger.error("[Generate Node] Error en _prioritize_relevant_snippets: {}", e)
            return full_context
    