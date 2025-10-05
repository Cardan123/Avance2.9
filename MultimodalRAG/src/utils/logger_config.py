import os
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from utils.config_file_manager import ConfigFileManager
import yaml

class LoggerConfig:
    """
    Logger configuration class using Loguru and Rich for enhanced logging.
    """

    @staticmethod
    def setup_logger(log_dir: str = None, log_file: str = "app.log"):
        """
        Set up the logger with specified log level and rich formatting.

        Args:
            log_dir (str): The directory where log files will be stored.
            log_file (str): The name of the log file.
        """
        # Si no se proporciona log_dir, usar ruta absoluta por defecto
        if log_dir is None:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            log_dir = os.path.join(project_root, "logs")
            
        #Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        logger.remove()  # Remove default logger
        logger.add(
            log_path, 
            rotation="10 MB",  # New log file is created when the current log file exceeds 10 MB
            retention="10 days", # Old log files are deleted after 10 days
            compression="zip", # Compress old log files to save space
            level="INFO", # Set log level to INFO
            format="{time: YYYY-MM-DD HH:mm:ss} {level} {message}"
        )
        console = Console()
        def safe_message(record):
            # Escapa llaves para evitar KeyError en Loguru
            msg = str(record['message']).replace('{', '{{').replace('}', '}}')
            return (
                  f"[{record['time']:%Y-%m-%d %H:%M:%S}] | <level>{record['level'].name}</level> | "
                f"{'[green]' if record['level'].name == 'INFO' else '[red]' if record['level'].name == 'ERROR' else '[yellow]' if record['level'].name == 'WARNING' else ''}"
                f"{msg}"
                f"{'[/green]' if record['level'].name == 'INFO' else '[/red]' if record['level'].name == 'ERROR' else '[/yellow]' if record['level'].name == 'WARNING' else ''}"
            )
        logger.add(
            RichHandler(console=console, markup=True, rich_tracebacks=True, show_time=True, show_level=True, show_path=False),
            level="INFO",
            format=safe_message
        )
        logger.info("Logger initialized and ready.")

        # Leer la configuración desde el archivo config_retrieval.yaml
        retrieval_config_yaml_path = ConfigFileManager.default_yaml_path()
        retrieval_config = ConfigFileManager.load_yaml_config(retrieval_config_yaml_path)

        # Obtener el valor de log_retrieval
        log_retrieval = retrieval_config.get("log_retrieval", True)

        # Configurar el logger dinámicamente
        if not log_retrieval:
            logger.remove()  # Eliminar todos los manejadores para deshabilitar completamente el logger
        else:
            logger.enable("__main__")  # Habilitar el logger si log_retrieval es True

""" if __name__ == "__main__":
    LoggerConfig.setup_logger()
    logger.info("This is a test log message.")
    logger.error("This is a test error message.") """