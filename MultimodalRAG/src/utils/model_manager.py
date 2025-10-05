import os
import json
import sys
from loguru import logger
from typing import List, Optional, Tuple, Any

# --- Ensure 'src' (parent of this folder) is on sys.path so 'utils' is importable ---
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSerializable  # type: ignore
from langchain_core.messages import AIMessage  # type: ignore
from utils.config_file_manager import ConfigFileManager
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer

# --- Vertex AI (opcional) ---
try:  # Estas importaciones solo funcionan si instalaste google-cloud-aiplatform
    import vertexai  # type: ignore
    from vertexai.generative_models import GenerativeModel  # type: ignore
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore
    service_account = None  # type: ignore

# Intentar importar la función externa si el usuario ya la creó
try:
    from auth.service_account import load_service_account_credentials  # type: ignore
except Exception:  # pragma: no cover
    # Fallback interno mínimo (si el usuario aún no definió su función). Puede eliminarse cuando exista la real.
    def load_service_account_credentials(key_path: str, export_env: bool = True, set_project_vars: bool = True, verbose: bool = True):
        import json
        from pathlib import Path
        if service_account is None:
            raise ImportError("Falta dependencia 'google-cloud-aiplatform' / 'google-auth'. Instala google-cloud-aiplatform.")
        p = Path(key_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No existe el archivo: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        required = {"type","project_id","private_key_id","private_key","client_email","client_id"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Faltan campos requeridos: {missing}")
        creds = service_account.Credentials.from_service_account_file(str(p))  # type: ignore
        project_id = data["project_id"]
        if export_env:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)
        if set_project_vars:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            os.environ["GCLOUD_PROJECT"] = project_id
        if verbose:
            logger.info(f"[Fallback load_service_account_credentials] project_id={project_id} path=..")
        return creds, project_id
# Device / dtype defaults
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

class BGETextEmbedder:
    def __init__(self, model_name="BAAI/bge-m3"):
        token = os.getenv("HUGGINGFACE_TOKEN")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            use_safetensors=True,
            trust_remote_code=True,
            token=token
        ).to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> List[List[float]]:
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)
            outputs = self.model(**tokens)
            # Mean pooling con máscara
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            mask = tokens.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.extend(emb.detach().cpu().tolist())
        return vecs

class BGEModelConfiguration:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model_name = model_name
        self.model, self.tokenizer = self._load_bge_model()

    def _load_bge_model(self):
        """
        Search for the BGE model in the local cache (snapshots) or download it if it doesn't exist.
        """
        from transformers import AutoModel, AutoTokenizer
        from utils.model_downloader import BGEModelDownloader
        token = os.getenv("HUGGINGFACE_TOKEN")
        downloader = BGEModelDownloader()
        snapshot_path = downloader.get_snapshot_path()
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        local_embedding_model_path = os.path.join(project_root, "src", "models", "BGE", "BAAI_bge-m3")
        if snapshot_path:
            logger.info(f"[BGEModelConfiguration] Loading BGE model FROM cache: {snapshot_path}")
            model = AutoModel.from_pretrained(snapshot_path)
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        else:
            logger.info(f"[BGEModelConfiguration] Downloading BGE model to {local_embedding_model_path}...")
            model = AutoModel.from_pretrained(self.model_name, cache_dir=local_embedding_model_path, use_auth_token=token)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=local_embedding_model_path, use_auth_token=token)
        return model, tokenizer
    
    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> List[List[float]]:
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)
            outputs = self.model(**tokens)
            # Mean pooling con máscara
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            mask = tokens.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.extend(emb.detach().cpu().tolist())
        return vecs

class RoBERTaModelConfiguration:
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self.model, self.tokenizer = self._load_roberta_model()

    def _load_roberta_model(self):
        """
        Search for the RoBERTa model in the local cache (snapshots) or download it if it doesn't exist.
        """
        from transformers import AutoModel, AutoTokenizer
        from utils.model_downloader import RoBERTaModelDownloader
        token = os.getenv("HUGGINGFACE_TOKEN")
        downloader = RoBERTaModelDownloader()
        snapshot_path = downloader.get_snapshot_path()
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        local_embedding_model_path = os.path.join(project_root, "src", "models", "RoBERTa", "facebookai_roberta-base")
        if snapshot_path:
            logger.info(f"[RoBERTaModelConfiguration] Loading RoBERTa model FROM cache: {snapshot_path}")
            # Replace AutoModelForMaskedLM with AutoModel to access hidden states
            model = AutoModel.from_pretrained(snapshot_path)
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        else:
            logger.info(f"[RoBERTaModelConfiguration] Downloading RoBERTa model to {local_embedding_model_path}...")
            model = AutoModel.from_pretrained(self.model_name, cache_dir=local_embedding_model_path, use_auth_token=token)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=local_embedding_model_path, use_auth_token=token)
        return model, tokenizer


class GeminiLLMWrapper:
    """
    Class to connect to Google Gemini and use the model.
    """
    def __init__(self):
        logger.info("[GeminiLLMWrapper] Initializing GeminiLLMWrapper...")
        load_dotenv()
        # Cargar config YAML
        self.retrieval_config_yaml_path = ConfigFileManager.default_yaml_path()
        self.retrieval_config = ConfigFileManager.load_yaml_config(self.retrieval_config_yaml_path)

        # Parámetros comunes
        self.model_name = self.retrieval_config.get("google_gemini_model_name", "gemini-1.5-pro")
        self.temperature = self.retrieval_config.get("google_gemini_temperature", 0.2)
        self.max_output_tokens = self.retrieval_config.get("google_gemini_max_output_tokens", 1024)
        self.convert_system_message_to_human = self.retrieval_config.get("google_gemini_convert_system_message_to_human", True)

        # Flags para decidir si usar Vertex AI con service account
        self.use_vertex_service_account = bool(self.retrieval_config.get(
            "google_gemini_use_vertex_service_account",
            os.getenv("GOOGLE_GEMINI_USE_VERTEX_SA", "false").lower() in ("1", "true", "yes")
        ))
        self.vertex_location = self.retrieval_config.get("google_gemini_vertex_location", os.getenv("GOOGLE_GEMINI_VERTEX_LOCATION", "us-central1"))
        self.vertex_service_account_json = self.retrieval_config.get(
            "google_gemini_service_account_json_path",
            os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        )

        self.llm = self._create_llm_unified()
        logger.info("[GeminiLLMWrapper] Initialization finished. Using VertexAI=%s" % self.use_vertex_service_account)

    def _create_llm_unified(self):
        """Decide qué backend usar (API key directa vs Vertex AI service account)."""
        if self.use_vertex_service_account:
            if not self.vertex_service_account_json:
                raise ValueError(
                    "Se habilitó 'google_gemini_use_vertex_service_account' pero no se definió 'google_gemini_service_account_json_path' ni GOOGLE_SERVICE_ACCOUNT_JSON."
                )
            if GenerativeModel is None:
                raise ImportError("No está instalado 'google-cloud-aiplatform' para usar Vertex AI.")
            logger.info(
                f"[GeminiLLMWrapper] Creando VertexAIGeminiClient model={self.model_name} location={self.vertex_location} sa=it's a secret."
            )
            client = VertexAIGeminiClient(
                service_account_json=self.vertex_service_account_json,
                model_name=self.model_name,
                location=self.vertex_location,
            )
            

            return VertexGeminiRunnable(inner=client, model_name=client.model_name)

        # Modo API key (LangChain Google Generative AI)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Falta GOOGLE_API_KEY y no se activó Vertex AI con service account. Configura uno de los dos métodos."
            )
        logger.info(f"[GeminiLLMWrapper] Creando ChatGoogleGenerativeAI model={self.model_name}")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            convert_system_message_to_human=self.convert_system_message_to_human,
            api_key=api_key,
        )

class VertexAIGeminiClient:
    """Cliente para usar modelos Gemini vía Vertex AI con service account.

    Usa la función `load_service_account_credentials` (propia del usuario) si existe.
    Caso contrario, usa un fallback interno simple.

    Ejemplo:
        client = VertexAIGeminiClient(
            service_account_json="secrets/mi-sa.json",
            model_name="gemini-2.5-flash",
            location="us-central1"
        )
        print(client.generate("Resume en 3 bullets la teoría de colas"))
    """

    def __init__(
        self,
        service_account_json: str,
        model_name: str = "gemini-2.5-flash",
        location: str = "us-central1",
        project_id: Optional[str] = None,
        auto_env: bool = True,
        verbose: bool = True,
    ) -> None:
        if GenerativeModel is None or vertexai is None:
            raise ImportError("[VertexAIGeminiClient] No se pudo importar Vertex AI. Instala 'google-cloud-aiplatform' en requirements.txt.")

        self.verbose = verbose
        self.location = location
        self.model_name = model_name
        self.project_id = project_id
        self._credentials = None
        self._model: Optional[GenerativeModel] = None  # type: ignore

        # Cargar credenciales usando la función del usuario (o fallback)
        creds, discovered_project = load_service_account_credentials(service_account_json)
        self._credentials = creds
        if not self.project_id:
            self.project_id = discovered_project

        # Inicializar Vertex AI
        if self.verbose:
            logger.info(f"[VertexAIGeminiClient] init project={self.project_id} location={self.location}")
        vertexai.init(project=self.project_id, location=self.location, credentials=self._credentials)
        self._model = GenerativeModel(self.model_name)  # type: ignore
        if self.verbose:
            logger.info(f"[VertexAIGeminiClient] Modelo listo: {self.model_name}")

        # Opcional: exportar variables entorno si se desea garantizar compatibilidad
        if auto_env:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.project_id or "")
            os.environ.setdefault("GCLOUD_PROJECT", self.project_id or "")

    # ---------------- API pública ---------------- #
    def generate(self, prompt: str, **kwargs) -> str:
        if not self._model:
            raise RuntimeError("Modelo no inicializado.")
        # Filtrar kwargs no soportados por el SDK de Gemini (p.ej. "return_exceptions" que LangChain puede propagar)
        # Mantén una pequeña lista blanca si en el futuro quieres permitir algunos parámetros.
        if kwargs:
            # Keys comunes que pueden llegar desde LangChain / runtime y que Gemini no acepta directamente
            unsupported = {
                "return_exceptions",
                "tags",
                "metadata",
                "run_name",
                "run_id",
                "stop",          # Gemini no usa 'stop' en generate_content
                "config",        # A veces aparece anidado
                "max_concurrency",
            }
            removed = []
            for k in list(kwargs.keys()):
                if k in unsupported:
                    kwargs.pop(k, None)
                    removed.append(k)
            if removed:
                logger.debug(f"[VertexAIGeminiClient.generate] Removed unsupported kwargs passed to Gemini: {removed}")

        try:
            resp = self._model.generate_content(prompt, **kwargs)  # type: ignore
        except TypeError as e:
            # Fallback: si aún falla por un kw nuevo/no contemplado, reintentar sin ninguno.
            logger.warning(f"[VertexAIGeminiClient.generate] TypeError with kwargs {list(kwargs.keys())}: {e}. Retrying without kwargs.")
            resp = self._model.generate_content(prompt)  # type: ignore
        text = getattr(resp, "text", None)
        if text is not None:
            return text
        # Fallback: concatenar candidates
        parts: List[str] = []
        for c in getattr(resp, "candidates", []) or []:  # type: ignore
            content = getattr(c, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:  # type: ignore
                    parts.append(getattr(p, "text", ""))
        return "\n".join([p for p in parts if p])

    def generate_full(self, prompt: str, **kwargs):
        if not self._model:
            raise RuntimeError("Modelo no inicializado.")
        if kwargs:
            unsupported = {
                "return_exceptions",
                "tags",
                "metadata",
                "run_name",
                "run_id",
                "stop",
                "config",
                "max_concurrency",
            }
            removed = []
            for k in list(kwargs.keys()):
                if k in unsupported:
                    kwargs.pop(k, None)
                    removed.append(k)
            if removed:
                logger.debug(f"[VertexAIGeminiClient.generate_full] Removed unsupported kwargs passed to Gemini: {removed}")
        try:
            return self._model.generate_content(prompt, **kwargs)  # type: ignore
        except TypeError as e:
            logger.warning(f"[VertexAIGeminiClient.generate_full] TypeError with kwargs {list(kwargs.keys())}: {e}. Retrying without kwargs.")
            return self._model.generate_content(prompt)  # type: ignore

    def count_tokens(self, prompt: str) -> int:
        if not self._model:
            return 0
        try:
            info = self._model.count_tokens([prompt])  # type: ignore
            return getattr(info, "total_tokens", 0)
        except Exception as e:  # pragma: no cover
            if self.verbose:
                logger.warning(f"[VertexAIGeminiClient] count_tokens error: {e}")
            return 0

    def reload(self, new_model_name: Optional[str] = None):
        if new_model_name:
            self.model_name = new_model_name
        if not self.project_id or not self._credentials:
            raise RuntimeError("No se pueden recargar sin project_id y credenciales.")
        vertexai.init(project=self.project_id, location=self.location, credentials=self._credentials)
        self._model = GenerativeModel(self.model_name)  # type: ignore
        if self.verbose:
            logger.info(f"[VertexAIGeminiClient] Modelo recargado: {self.model_name}")

    # ---------------- Getters ---------------- #
    @property
    def credentials(self):
        return self._credentials

    @property
    def model(self):  # Devuelve la instancia del SDK
        return self._model
    
# Adaptador Runnable para integrarse con LangChain LLMChain
class VertexGeminiRunnable(RunnableSerializable):  # type: ignore
    """Adaptador RunnableSerializable para VertexAIGeminiClient.

    Definimos los campos como anotaciones para que Pydantic (usado por LangChain)
    los registre y no falle al acceder a model_name.
    """
    inner: Any
    model_name: str

    # Permitir tipos arbitrarios (VertexAIGeminiClient)
    model_config = {"arbitrary_types_allowed": True}

    def _coerce_prompt(self, value: Any) -> str:
        """Best effort conversion of arbitrary LangChain / SDK message-like inputs to a plain text prompt.

        Handles cases where message.content is a list (multimodal: text + image parts) to avoid
        TypeError when joining non-string elements (e.g. dicts for images).
        """

        def _content_to_string(content: Any) -> str:
            # Direct string
            if isinstance(content, str):
                return content
            # Bytes -> decode
            if isinstance(content, (bytes, bytearray)):
                try:
                    return content.decode("utf-8", errors="ignore")
                except Exception:
                    return str(content)
            # List of parts (could be str, dict, nested lists)
            if isinstance(content, list):
                subparts: List[str] = []
                for part in content:
                    # Recursively normalize
                    subparts.append(_content_to_string(part))
                return "\n".join([p for p in subparts if p])
            # Dict: try common text keys, else stringify
            if isinstance(content, dict):
                for key in ("text", "content", "prompt", "input", "question"):
                    val = content.get(key)
                    if val:
                        return _content_to_string(val)
                # Multimodal Gemini style: {"type": "text", "text": "..."} or image blocks
                if content.get("type") == "text" and content.get("text"):
                    return _content_to_string(content["text"])
                # Skip pure image parts but leave a lightweight marker if url present
                if content.get("type") in ("image", "image_url", "media"):
                    url = content.get("url") or (content.get("image_url") or {}).get("url")
                    return f"[IMAGE:{url or 'inline'}]"
                # Fallback: stable deterministic ordering
                try:
                    return json.dumps(content, ensure_ascii=False)
                except Exception:
                    return str(content)
            # Any other object: look for .text / .content
            for attr in ("text", "content"):
                if hasattr(content, attr):
                    try:
                        extracted = getattr(content, attr)
                        if extracted:
                            return _content_to_string(extracted)
                    except Exception:
                        pass
            return str(content)

        # Dict at top-level
        if isinstance(value, dict):
            for k in ("prompt", "input", "question", "text"):
                if k in value and value[k]:
                    return _content_to_string(value[k])
            # Concatenate remaining values
            return "\n".join(
                _content_to_string(v) for v in value.values() if v is not None
            )

        # Sequence of messages / parts
        if isinstance(value, list):
            parts: List[str] = []
            for msg in value:
                content = (
                    getattr(msg, "content", None)
                    or getattr(msg, "text", None)
                    or msg
                )
                text_part = _content_to_string(content)
                if text_part:
                    parts.append(text_part)
            return "\n".join(parts)

        return _content_to_string(value)

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs):  # type: ignore
        prompt = self._coerce_prompt(input)
        text = self.inner.generate(prompt, **kwargs)
        return AIMessage(content=text)

    def batch(self, inputs: List[Any], config: Optional[dict] = None, **kwargs):  # type: ignore
        return [self.invoke(i, config=config, **kwargs) for i in inputs]

    def stream(self, input: Any, config: Optional[dict] = None, **kwargs):  # type: ignore
        yield self.invoke(input, config=config, **kwargs)

    def count_tokens(self, input: Any) -> int:  # type: ignore
        prompt = self._coerce_prompt(input)
        try:
            return self.inner.count_tokens(prompt)
        except Exception:
            return 0




if __name__ == "__main__":
    BGEEmbdider = BGETextEmbedder()
    texts = ["Hola, ¿cómo estás?", "Este es un ejemplo de incrustaciónón de texto."]
    embeddings = BGEEmbdider.encode(texts)
    print(embeddings) 