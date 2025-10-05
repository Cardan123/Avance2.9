import os
from fastmcp import FastMCP
from common.logging_setup import setup_logging
from tools.image_pipeline_tool import run_image_pipeline

# Configuración del servidor
SERVER_NAME = "NewMCPServer"
LOG_FILE_NAME = f"{SERVER_NAME.lower()}_debug.log"
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", LOG_FILE_NAME)

# Configurar logging
logger, debug_log_file = setup_logging(SERVER_NAME, log_file_path)

# Crear el servidor MCP
mcp = FastMCP(SERVER_NAME)

@mcp.tool()
def run_pipeline() -> dict:
    return run_image_pipeline()

if __name__ == "__main__":
    debug_log_file.write(f"Launching {SERVER_NAME} with STDIO transport\n")
    debug_log_file.flush()
    try:
        mcp.run(transport="stdio")
    finally:
        debug_log_file.close()