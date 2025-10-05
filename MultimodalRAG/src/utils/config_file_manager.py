import os
import yaml
from loguru import logger

class ConfigFileManager:
    """
    Clase para gestionar la lectura y escritura de archivos de configuración en formato YAML.
    """
    @staticmethod
    def default_yaml_path():
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "config_retrieval.yaml")
    
    @staticmethod
    def load_yaml_config(path):
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"[ConfigFileManager] Loaded YAML config from {path}")
            return config
        except Exception as e:
            logger.error(f"[ConfigFileManager] Error loading YAML config from {path}: {e}")
            return {}

    @staticmethod
    def load_vector_search_template_json(path):
        try:
            with open(path, "r") as f:
                pipeline = f.read()
            logger.info(f"[ConfigFileManager] Loaded vector search template from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"[ConfigFileManager] Error loading vector search template from {path}: {e}")
            return ""
        
    @staticmethod
    def load_prompt_template_json(path):
        try:
            with open(path, "r") as f:
                template = f.read()
            logger.info(f"[ConfigFileManager] Loaded prompt template from {path}")
            return template
        except Exception as e:
            logger.error(f"[ConfigFileManager] Error loading prompt template from {path}: {e}")
            return ""