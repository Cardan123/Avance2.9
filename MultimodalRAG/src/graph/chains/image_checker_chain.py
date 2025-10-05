from __future__ import annotations

"""CheckerChain

Encapsula la lógica del 'checker' que analiza descripciones de imágenes
para determinar si alguna responde (total o parcialmente) a la pregunta.

Responsabilidades:
  - Construir el checker_context desde el estado (imagenes + descripciones JSON)
  - Cargar el prompt template configurado (checker_prompt_template_path)
  - Invocar el LLM y parsear la salida estricta JSON (lista)
  - Exponer un método run() que devuelve (matches, checker_context)

La salida esperada del LLM es una LISTA JSON de objetos con claves:
  filename, relevance ('full'|'partial'|'none'), confidence (0..1),
  justification, extracted_answer
"""

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class ImageCheckerChain:
    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self._chain: Optional[LLMChain] = None

    # ------------------------------ Public API ------------------------------ #
    def run(self, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        """Ejecuta el checker: construye contexto, invoca LLM y parsea.

        Devuelve (matches, checker_context)
        matches: lista de dicts parseados de la respuesta del modelo.
        """
        checker_context = state["images_context"] or ""
        if not checker_context.strip():
            logger.warning("[CheckerChain] checker_context vacío; se retorna sin invocar LLM.")
            return [], checker_context

        chain = self._get_or_create_chain()
        if chain is None:
            logger.warning("[CheckerChain] No se pudo crear chain del checker.")
            return [], checker_context

        question = state.get("question", "")
        try:
            raw = chain.run(question=question, checker_context=checker_context)
        except Exception as e:
            logger.error("[CheckerChain] Error invocando LLM: {}", e)
            return [], checker_context

        matches = self._parse_checker_output(raw)
        logger.info("[CheckerChain] {} matches parseados.", len(matches))
        return matches, checker_context

    # ---------------------------- Internal Helpers ------------------------- #
    def _get_or_create_chain(self) -> Optional[LLMChain]:
        if self._chain is not None:
            return self._chain
        try:
            config = self.retriever.vector_search_pipeline.vector_search_config.retrieval_config
            checker_path = config.get("checker_prompt_template_path")
            if not checker_path or not os.path.isfile(checker_path):
                logger.warning("[CheckerChain] Archivo de prompt checker no encontrado: {}", checker_path)
                return None
            with open(checker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            template = data.get("template", "")
            input_vars = data.get("input_variables", ["question", "checker_context"])  # fallback
            prompt = PromptTemplate(template=template, input_variables=input_vars)
            llm = self.retriever.gemini_llm_wrapper.llm
            self._chain = LLMChain(prompt=prompt, llm=llm)
            return self._chain
        except Exception as e:
            logger.error("[CheckerChain] Error creando chain: {}", e)
            return None

    @staticmethod
    def _parse_checker_output(raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        txt = raw.strip()
        try:
            start = txt.index('[')
            end = txt.rindex(']') + 1
            txt = txt[start:end]
        except ValueError:
            pass
        try:
            data = json.loads(txt)
            return data if isinstance(data, list) else []
        except Exception:
            return []

