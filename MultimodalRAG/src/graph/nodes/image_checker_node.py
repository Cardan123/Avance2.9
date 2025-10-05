from loguru import logger
import os
from typing import List, Dict, Any
from graph.chains.image_checker_chain import ImageCheckerChain


class ImageCheckerNode:
    @staticmethod
    def process(state, retriever=None):
        logger.info("[Checker Node] Starting checker process.")
        if retriever is None:
            logger.warning("[Checker Node] Retriever not provided; skipping.")
            return state

        chain = ImageCheckerChain(retriever)
        matches, checker_context = chain.run(state)
        state["checker_context"] = checker_context

        answer_paths: List[str] = []
        answer_contents: List[str] = []
        answer_justifications: List[str] = []
        if matches:
            image_paths = state.get("image_paths") or []
            paths_by_name = {os.path.basename(p): p for p in image_paths}
            for m in matches:
                if not isinstance(m, dict):
                    continue
                relevance = (m.get("relevance") or "none").lower()
                if relevance in ("full", "partial"):
                    fname = m.get("filename")
                    extracted_answer = m.get("extracted_answer") or ""
                    extracted_justification = m.get("justification") or ""
                    if fname and fname in paths_by_name:
                        answer_paths.append(paths_by_name[fname])
                        answer_contents.append(extracted_answer)
                        answer_justifications.append(extracted_justification)

        if answer_paths:
            state["images_has_answers"] = True
        state["answer_image_paths"] = answer_paths
        state["answer_image_contents"] = answer_contents
        state["answer_image_justifications"] = answer_justifications
        logger.info(
            "[Checker Node] images_has_answers={} matched_paths={}",
            state.get("images_has_answers"),
            len(answer_paths),
        )
        return state