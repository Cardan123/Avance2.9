"""
Final test with gemini-2.0-flash-exp
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from retrieval.image_description_chain import ImageDescriptionChain
from utils.model_manager import GeminiLLMWrapper
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=''), colorize=True, level="INFO")

load_dotenv()

print("="*60)
print("FINAL TEST - GEMINI 2.0 FLASH EXP")
print("="*60)

# Find test image
test_images = [
    r"C:\Git\MultimodalRAG\data\images\segments\pig_2\segment_0.png",
    r"C:\Git\MultimodalRAG\data\images\pig_2.jpg"
]

test_image = None
for img_path in test_images:
    if os.path.exists(img_path):
        test_image = img_path
        break

if not test_image:
    print("❌ No test image found!")
    # Search for any image
    data_dir = r"C:\Git\MultimodalRAG\data"
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(root, file)
                    print(f"Found image: {test_image}")
                    break
            if test_image:
                break

if not test_image:
    print("❌ Cannot find any image to test!")
    exit(1)

print(f"\n✓ Using test image: {test_image}")

# Initialize
print("\n" + "="*60)
print("Initializing components...")
print("="*60)

try:
    gemini_wrapper = GeminiLLMWrapper()
    print(f"✓ LLM initialized: {gemini_wrapper.model_name}")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    chain = ImageDescriptionChain(llm=gemini_wrapper.llm)
    print("✓ Chain initialized")
except Exception as e:
    print(f"❌ Failed to initialize chain: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test
print("\n" + "="*60)
print("Describing image...")
print("="*60)

result = chain.describe_single(test_image, extra_instruction="")

print("\n" + "="*60)
print("RESULT:")
print("="*60)

if result.get("empty_response"):
    print("\n❌ EMPTY RESPONSE!")
    print(f"Error: {result.get('error', 'No error info')}")
    print(f"Finish reasons: {result.get('finish_reasons', [])}")
    print(f"Safety blocked: {result.get('safety_blocked', False)}")
    print(f"Safety info: {result.get('safety', [])}")
else:
    print(f"\n✅ SUCCESS!")
    print(f"\nRaw description ({len(result['raw_description'])} chars):")
    print(result['raw_description'])
    
    if result.get('parsed'):
        print(f"\n✓ Parsed JSON:")
        import json
        print(json.dumps(result['parsed'], indent=2, ensure_ascii=False))
    else:
        print("\n⚠️ Could not parse as JSON (but got text response)")

print("\n" + "="*60)
print("If this works, your ImageDescriptionChain is fixed!")
print("="*60)
