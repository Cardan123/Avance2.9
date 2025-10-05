# ingest_pipeline_mcp_server.py  (modo STDIO)

# --- IMPORTANTE: proteger el canal STDIO antes de cualquier import pesado ---
import sys
import logging
import builtins

# 1) Mantén stdout LIMPIO (no lo redirijas). En su lugar,
#    manda cualquier print(...) a stderr:
_orig_print = builtins.print
def _print_to_stderr(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    return _orig_print(*args, **kwargs)
builtins.print = _print_to_stderr

# 2) Configura logging a stderr (no a stdout)
logger = logging.getLogger("IngestPipelineMCPServer")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

stderr_handler = logging.StreamHandler()  # stderr por defecto
stderr_handler.setLevel(logging.DEBUG)
stderr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(stderr_handler)

# (Opcional) archivo de log
file_handler = logging.FileHandler("ingestPipelineMCPServer.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

logger.debug("Starting IngestPipelineMCPServer (STDIO mode)")

# --- resto de imports ya seguros (si hacen print, se va a stderr) ---
from fastmcp import FastMCP

# Cargar variables de entorno desde .env
import os
from pathlib import Path

# Buscar el archivo .env en el directorio del proyecto
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"

if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    logger.debug(f"Loaded environment variables from {env_file}")
else:
    logger.warning(f"No .env file found at {env_file}")

try:
    from utils.logger_config import LoggerConfig
    LoggerConfig.setup_logger()
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
        logger.debug("Calling ImageIngestPipeline().run()")
        pipeline = ImageIngestPipeline()
        pipeline.run()  # cualquier print del pipeline irá a stderr (seguro)
        logger.debug("Pipeline execution completed successfully.")
        return {"status": "success", "message": "Pipeline ran successfully."}
    except Exception as e:
        logger.exception("Error running pipeline")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.debug("Launching MCP server with STDIO transport")
    # MODO STDIO: NO redirijas stdout; FastMCP lo usa para JSON-RPC
    mcp.run(transport="stdio")
