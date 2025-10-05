from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from loguru import logger
from langchain_core.messages import HumanMessage
from utils.config_file_manager import ConfigFileManager
from utils.model_manager import GeminiLLMWrapper
from .prompt_template_creator import PromptTemplateCreator
from langchain.prompts import PromptTemplate
from utils.image_utils import image_file_to_base64, build_data_url


class ImageDescriptionPromptLoader:
    """Loads the image description prompt template (JSON) and produces a PromptTemplate."""

    def __init__(self):
        self.config = ConfigFileManager.load_yaml_config(ConfigFileManager.default_yaml_path())
        self.path = self.config.get(
            "image_description_prompt_template_path",
            # fallback relative path
            str(Path(__file__).parent / "prompt_templates" / "image_description_prompt.json")
        )
        self.template, self.input_variables = self._load_json(self.path)
        self.prompt_template = PromptTemplate(template=self.template, input_variables=self.input_variables)

    def _load_json(self, file_path: str):
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("template", ""), data.get("input_variables", [])


class ImageDescriptionChain:
    """Chain that takes image paths and produces structured descriptions using Gemini multimodal."""

    def __init__(self, llm=None, prompt: Optional[PromptTemplate] = None):
        self.llm = llm or GeminiLLMWrapper().llm
        if prompt is None:
            self.prompt_loader = ImageDescriptionPromptLoader()
            self.prompt = self.prompt_loader.prompt_template
        else:
            self.prompt = prompt

    def _build_message(self, image_path: str, extra_instruction: str = ""):
        filename = Path(image_path).name
        try:
            mime_subtype, b64 = image_file_to_base64(image_path)
        except Exception as e:
            return None, {"path": image_path, "error": str(e)}

        # Render textual instruction using prompt template with error handling
        try:
            rendered_instruction = self.prompt.format(filename=filename, extra_instruction=extra_instruction)
        except KeyError as e:
            logger.error(f"[ImageDescriptionChain] Missing variable in prompt template: {e}. Using fallback template.")
            rendered_instruction = (
                "Analiza la imagen y devuelve un JSON con claves: resumen, objetos, texto, utilidad, detalle_adicional, archivo. "
                f"Usa el nombre de archivo '{filename}'. Instrucción extra: {extra_instruction}"
            )

        content = [
            {"type": "text", "text": rendered_instruction},
            {"type": "image_url", "image_url": {"url": build_data_url(mime_subtype, b64)}},
        ]
        return HumanMessage(content=content), None

    def describe_single(self, image_path: str, extra_instruction: str = "") -> Dict[str, Any]:
        message, error = self._build_message(image_path, extra_instruction)
        if error:
            return error
        try:
            # Debug info before sending
            try:
                img_part = next((p for p in message.content if isinstance(p, dict) and p.get("type") == "image_url"), None)
                b64_len = len(img_part.get("image_url", {}).get("url", "")) if img_part else 0
                logger.debug(f"[ImageDescriptionChain] Invoking LLM for '{image_path}' (b64 url length={b64_len})")
            except Exception:
                pass

            response = self.llm.invoke([message])  # ChatGoogleGenerativeAI expects list of messages

            logger.debug(f"[ImageDescriptionChain] Raw response object type={type(response)} repr={response!r}")

            # Some providers return AIMessage with content either a str or list[dict]
            raw_content = getattr(response, "content", response)

            def _extract_text(content):
                # If already string
                if isinstance(content, str):
                    return content
                # LangChain multimodal may return list[{'type': 'text', 'text': '...'}, ...]
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text" and part.get("text"):
                                texts.append(part["text"])
                            # Gemini sometimes returns just {'text': '...'}
                            elif "text" in part and isinstance(part["text"], str):
                                texts.append(part["text"])
                        elif isinstance(part, str):
                            texts.append(part)
                    return "\n".join(t.strip() for t in texts if t.strip())
                # Fallback: try standard attribute
                if hasattr(content, "text") and isinstance(getattr(content, "text"), str):
                    return content.text  # type: ignore
                return str(content)

            cleaned = _extract_text(raw_content).strip()

            # Try parse JSON snippet if present
            parsed_json = None
            try:
                # Heuristic: find first '{' and last '}'
                if "{" in cleaned and "}" in cleaned:
                    start = cleaned.index("{")
                    end = cleaned.rindex("}") + 1
                    json_segment = cleaned[start:end]
                    parsed_json = json.loads(json_segment)
            except Exception:
                parsed_json = None

            return {
                "path": image_path,
                "filename": Path(image_path).name,
                "raw_description": cleaned,
                "parsed": parsed_json,
                "empty_response": not bool(cleaned),
                "provider_raw": raw_content if isinstance(raw_content, str) else None
            }
        except Exception as e:
            logger.exception(f"[ImageDescriptionChain] Error invoking LLM for {image_path}: {e}")
            return {"path": image_path, "error": str(e)}

    def run(self, image_paths: List[str], extra_instruction: str = "") -> List[Dict[str, Any]]:
        results = []
        for p in image_paths:
            results.append(self.describe_single(p, extra_instruction=extra_instruction))
        return results


def build_image_description_chain(llm=None) -> ImageDescriptionChain:
    return ImageDescriptionChain(llm=llm)
