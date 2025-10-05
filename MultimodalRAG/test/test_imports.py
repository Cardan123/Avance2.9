#!/usr/bin/env python3
"""
Script de prueba para verificar imports
"""
import sys
import os

# Agregar el directorio src al path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")

try:
    print("Testing basic imports...")
    import yaml
    print("✓ yaml imported successfully")
    
    from pathlib import Path
    print("✓ pathlib imported successfully")
    
    from fastmcp import FastMCP
    print("✓ fastmcp imported successfully")
    
    from utils.logger_config import LoggerConfig
    print("✓ LoggerConfig imported successfully")
    
    from ingest.image_ingest_pipeline import ImageIngestPipeline
    print("✓ ImageIngestPipeline imported successfully")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"❌ Error importing: {e}")
    import traceback
    traceback.print_exc()
