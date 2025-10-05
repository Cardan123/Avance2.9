"""
Test script for ImageDescriptionChain to diagnose empty responses
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from retrieval.image_description_chain import ImageDescriptionChain
from utils.model_manager import GeminiLLMWrapper
from loguru import logger

# Configure logger
logger.remove()
logger.add(lambda msg: print(msg, end=''), colorize=True, level="DEBUG")

load_dotenv()

def test_image_description():
    """Test image description with detailed logging"""
    
    # Path de una imagen de prueba - AJUSTA ESTA RUTA
    test_image_path = r"C:\Git\MultimodalRAG\data\images\segments\pig_2\segment_0.png"
    
    # Verificar que existe
    if not os.path.exists(test_image_path):
        logger.error(f"Image not found: {test_image_path}")
        logger.info("Available images in segments:")
        segments_dir = r"C:\Git\MultimodalRAG\data\images\segments"
        if os.path.exists(segments_dir):
            for root, dirs, files in os.walk(segments_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        logger.info(f"  - {os.path.join(root, file)}")
        return
    
    logger.info("="*60)
    logger.info("Testing ImageDescriptionChain")
    logger.info("="*60)
    
    # Test 1: Verificar API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment!")
        return
    logger.success(f"✓ API Key found: {api_key[:10]}...")
    
    # Test 2: Crear instancia del LLM
    logger.info("\nInitializing Gemini LLM...")
    try:
        gemini_wrapper = GeminiLLMWrapper()
        logger.success(f"✓ LLM initialized: {gemini_wrapper.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Crear chain
    logger.info("\nInitializing ImageDescriptionChain...")
    try:
        chain = ImageDescriptionChain(llm=gemini_wrapper.llm)
        logger.success("✓ Chain initialized")
    except Exception as e:
        logger.error(f"Failed to initialize chain: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Probar con imagen
    logger.info(f"\nDescribing image: {test_image_path}")
    logger.info("-"*60)
    
    result = chain.describe_single(test_image_path, extra_instruction="")
    
    logger.info("\n" + "="*60)
    logger.info("RESULT:")
    logger.info("="*60)
    
    # Mostrar resultado detallado
    for key, value in result.items():
        if key == "raw_description":
            logger.info(f"\n{key}:")
            logger.info(f"  Length: {len(value) if value else 0}")
            logger.info(f"  Content: {value if value else '(EMPTY)'}")
        elif key == "parsed":
            logger.info(f"\n{key}: {value}")
        elif key == "empty_response":
            if value:
                logger.error(f"\n⚠ {key}: {value}")
            else:
                logger.success(f"\n✓ {key}: {value}")
        elif key in ["finish_reasons", "safety", "safety_blocked"]:
            if value:
                logger.warning(f"\n{key}: {value}")
        else:
            logger.info(f"{key}: {value}")
    
    # Test 5: Prueba directa sin imagen (solo texto)
    logger.info("\n" + "="*60)
    logger.info("Testing LLM without image (text only):")
    logger.info("="*60)
    
    try:
        from langchain_core.messages import HumanMessage
        simple_msg = HumanMessage(content="Responde solo: 'Test OK'")
        response = gemini_wrapper.llm.invoke([simple_msg])
        logger.success(f"✓ Text-only response: {response.content}")
    except Exception as e:
        logger.error(f"Text-only test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_description()
