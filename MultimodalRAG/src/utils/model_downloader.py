import os
from transformers import CLIPModel, CLIPProcessor
from loguru import logger
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dotenv import load_dotenv

class BGEModelDownloader:
    """
    Utility class to download and cache BGE model and tokenizer locally.
    """
    def __init__(self, model_name="BAAI/bge-m3", base_dir=None):
        self.model_name = model_name
        # Si no se proporciona base_dir, usar ruta absoluta por defecto
        if base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_dir = os.path.join(project_root, "src", "models")
        
        self.base_dir = base_dir
        self.model_dir = os.path.join(base_dir, "BGE")
        os.makedirs(self.model_dir, exist_ok=True)

    def get_model(self, use_auth_token=None):
        """
        Download BGE model and tokenizer if not present locally, else load from local cache.
        Returns: (AutoModel, AutoTokenizer)
        """
        snapshot_path = self.get_snapshot_path()
        local_model_path = os.path.join(self.model_dir, self.model_name.replace('/', '_'))
        if snapshot_path:
            logger.info(f"[BGEModelDownloader] Loading BGE model from cache: {snapshot_path}")
            model = AutoModelForMaskedLM.from_pretrained(snapshot_path)
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        else:
            logger.info(f"[BGEModelDownloader] Downloading BGE model to {local_model_path}...")
            model = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=local_model_path, use_auth_token=use_auth_token)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=local_model_path, use_auth_token=use_auth_token)
            logger.info(f"[BGEModelDownloader] Model downloaded and cached.")
        return model, tokenizer

    def get_snapshot_path(self):
        """
        Verifies and returns the path of a valid snapshot if it exists, None if it doesn't.
        """
        logger.info(f"[BGEModelDownloader] Checking for cached snapshots...")
        local_model_path = os.path.join(self.model_dir, self.model_name.replace('/', '_'))
        hf_subfolder = os.path.join(local_model_path, "models--baai--bge-m3")
        snapshots_dir = os.path.join(hf_subfolder, "snapshots")
        if os.path.exists(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, name)
                if os.path.isdir(candidate):
                    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json"]
                    files_exist = all(os.path.exists(os.path.join(candidate, f)) for f in required_files)
                    if files_exist:
                        logger.info(f"[BGEModelDownloader] Found valid snapshot: {candidate}")
                        return candidate
                    
        logger.info(f"[BGEModelDownloader] No valid snapshots found.")
        return None
    
class RoBERTaModelDownloader:
    """
    Utility class to download and cache RoBERTa model and tokenizer locally.
    """
    def __init__(self, model_name="FacebookAI/roberta-base", base_dir=None):
        self.model_name = model_name
        # Si no se proporciona base_dir, usar ruta absoluta por defecto
        if base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_dir = os.path.join(project_root, "src", "models")
        
        self.base_dir = base_dir
        self.model_dir = os.path.join(base_dir, "RoBERTa")
        os.makedirs(self.model_dir, exist_ok=True)

    def get_model(self, use_auth_token=None):
        """
        Download RoBERTa model and tokenizer if not present locally, else load from local cache.
        Returns: (AutoModelForMaskedLM, AutoTokenizer)
        """
        snapshot_path = self.get_snapshot_path()
        local_model_path = os.path.join(self.model_dir, self.model_name.replace('/', '_'))
        if snapshot_path:
            logger.info(f"[RoBERTaModelDownloader] Loading RoBERTa model from cache: {snapshot_path}")
            model = AutoModelForMaskedLM.from_pretrained(snapshot_path)
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        else:
            logger.info(f"[RoBERTaModelDownloader] Downloading RoBERTa model to {local_model_path}...")
            model = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=local_model_path, use_auth_token=use_auth_token)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=local_model_path, use_auth_token=use_auth_token)
            logger.info(f"[RoBERTaModelDownloader] Model downloaded and cached.")
        return model, tokenizer

    def get_snapshot_path(self):
        """
        Verifies and returns the path of a valid snapshot if it exists, None if it doesn't.
        """
        logger.info(f"[RoBERTaModelDownloader] Checking for cached snapshots...")
        local_model_path = os.path.join(self.model_dir, self.model_name.replace('/', '_'))
        hf_subfolder = os.path.join(local_model_path, "models--facebookai--roberta-base")
        snapshots_dir = os.path.join(hf_subfolder, "snapshots")
        if os.path.exists(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, name)
                if os.path.isdir(candidate):
                    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
                    files_exist = all(os.path.exists(os.path.join(candidate, f)) for f in required_files)
                    if files_exist:     
                        logger.info(f"[RoBERTaModelDownloader] Found valid snapshot: {candidate}")
                        return candidate
                    
        logger.info(f"[RoBERTaModelDownloader] No valid snapshots found.")
        return None
    
class CLIPModelDownloader:

    """
    Utility class to download and cache CLIP model and processor locally.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16", base_dir=None):
        self.model_name = model_name
        # Si no se proporciona base_dir, usar ruta absoluta por defecto
        if base_dir is None:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_dir = os.path.join(project_root, "src", "models")
        
        self.base_dir = base_dir
        self.model_dir = os.path.join(base_dir, "CLIP")
        os.makedirs(self.model_dir, exist_ok=True)

    def get_model(self, use_auth_token=None):
        """
        Download CLIP model and processor if not present locally, else load from local cache.
        Returns: (CLIPModel, CLIPProcessor)
        """
        snapshot_path = self.get_snapshot_path()
        local_model_path = os.path.join(self.model_dir, self.model_name.replace('/', '_'))
        if snapshot_path:
            logger.info(f"[CLIPModelDownloader] Loading CLIP model from cache: {snapshot_path}")
            model = CLIPModel.from_pretrained(snapshot_path)
            processor = CLIPProcessor.from_pretrained(snapshot_path)
        else:
            logger.info(f"[CLIPModelDownloader] Downloading CLIP model to {local_model_path}...")
            model = CLIPModel.from_pretrained(self.model_name, cache_dir=local_model_path, use_auth_token=use_auth_token)
            processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=local_model_path, use_auth_token=use_auth_token)
            logger.info(f"[CLIPModelDownloader] Model downloaded and cached.")
        return model, processor

    def get_snapshot_path(self):
        """
        Verifies and returns the path of a valid snapshot if it exists, None if it doesn't.
        """
        local_model_path = os.path.join(self.model_dir, self.model_name.replace('/', '_'))
        hf_subfolder = os.path.join(local_model_path, "models--openai--clip-vit-base-patch16")
        snapshots_dir = os.path.join(hf_subfolder, "snapshots")
        if os.path.exists(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, name)
                if os.path.isdir(candidate):
                    required_files = ["pytorch_model.bin", "config.json"]
                    files_exist = all(os.path.exists(os.path.join(candidate, f)) for f in required_files)
                    if files_exist:
                        return candidate
        return None


  
if __name__ == "__main__":
    import os
    # Carga variable de entorno para el token
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    downloader = RoBERTaModelDownloader()
    model, tokenizer = downloader.get_model(use_auth_token=token)
    print("RoBERTa model and tokenizer are ready and cached.")
