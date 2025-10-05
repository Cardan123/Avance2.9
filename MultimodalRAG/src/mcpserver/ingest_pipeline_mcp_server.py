# ingest_pipeline_mcp_server.py  (modo STDIO) - VERSION MEJORADA

# --- IMPORTANTE: proteger el canal STDIO antes de cualquier import pesado ---
import sys
import logging
import builtins
import warnings
import os

# Suprimir todas las advertencias
warnings.filterwarnings("ignore")

log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "mcp_server_debug.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

debug_log_file = open(log_file_path, "w", encoding="utf-8")

_orig_print = builtins.print
def _print_to_logfile(*args, **kwargs):
    kwargs.setdefault("file", debug_log_file)
    return _orig_print(*args, **kwargs)
builtins.print = _print_to_logfile

logger = logging.getLogger("IngestPipelineMCPServer")
logger.setLevel(logging.WARNING)  # Solo errores graves
logger.handlers.clear()

file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("fastmcp").setLevel(logging.ERROR)

class StderrCapture:
    def __init__(self, original_stderr, log_file):
        self.original_stderr = original_stderr
        self.log_file = log_file
    
    def write(self, text):
        # Solo envía a log file, NO a stderr original
        self.log_file.write(text)
        self.log_file.flush()
    
    def flush(self):
        self.log_file.flush()

sys.stderr = StderrCapture(sys.stderr, debug_log_file)

logger.warning("Starting IngestPipelineMCPServer (STDIO mode) - Enhanced error handling")

from fastmcp import FastMCP

from pathlib import Path

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"

if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

try:
    import sys
    sys.path.append(str(project_root / "src"))
    from utils.logger_config import LoggerConfig
    # NO llamar setup_logger() para evitar conflictos
    pass
except Exception:
    pass

from ingest.image_ingest_pipeline import ImageIngestPipeline  # noqa: E402

# Crea el servidor MCP
mcp = FastMCP("IngestPipelineMCPServer")

@mcp.tool()
def run_ingest_pipeline() -> dict:
    """
    Run the image ingestion pipeline.
    """
    try:
        # Capturar CUALQUIER output del pipeline
        pipeline = ImageIngestPipeline()
        
        # Ejecutar el pipeline en un contexto silencioso
        import contextlib
        import io
        
        # Crear buffer para capturar cualquier output residual
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output), \
             contextlib.redirect_stderr(captured_output):
            pipeline.run()
        
        # Log solo lo esencial al archivo
        debug_log_file.write(f"Pipeline execution completed successfully at {__import__('datetime').datetime.now()}\n")
        debug_log_file.flush()
        
        return {"status": "success", "message": "Pipeline ran successfully."}
        
    except Exception as e:
        # Log error solo al archivo
        debug_log_file.write(f"Pipeline error: {str(e)}\n")
        debug_log_file.flush()
        return {"status": "error", "message": f"Pipeline failed: {str(e)}"}

if __name__ == "__main__":
    debug_log_file.write("Launching MCP server with STDIO transport\n")
    debug_log_file.flush()
    
    try:
        # MODO STDIO: FastMCP usa stdout para JSON-RPC
        mcp.run(transport="stdio")
    finally:
        # Cerrar archivo de log al terminar
        debug_log_file.close()
