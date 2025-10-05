"""
Test the direct Gemini wrapper
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from utils.model_manager_direct import GeminiLLMWrapperDirect
from langchain_core.messages import HumanMessage

load_dotenv()

print("="*60)
print("TESTING DIRECT GEMINI WRAPPER")
print("="*60)

# Initialize wrapper
try:
    wrapper = GeminiLLMWrapperDirect()
    print(f"✓ Wrapper initialized with model: {wrapper.model_name}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    exit(1)

# Test 1: Simple text
print("\n" + "="*60)
print("Test 1: Simple text")
print("="*60)

try:
    response = wrapper.invoke("Say 'Direct wrapper OK'")
    print(f"✓ Response: {response.content}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: With image
print("\n" + "="*60)
print("Test 2: Image description")
print("="*60)

test_images = [
    r"C:\Git\MultimodalRAG\data\images\segments\pig_2\segment_0.png",
    r"C:\Git\MultimodalRAG\data\images\pig_2.jpg"
]

test_image = None
for img_path in test_images:
    if os.path.exists(img_path):
        test_image = img_path
        break

if test_image:
    print(f"Using image: {test_image}")
    
    import base64
    from PIL import Image
    from io import BytesIO
    
    # Load and convert image
    with Image.open(test_image) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 1024:
            scale = 1024 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Create message
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    )
    
    try:
        response = wrapper.invoke([message])
        print(f"✓ Response: {response.content}")
        print("\n✅✅ SUCCESS! Direct wrapper works with vision!")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ No test image found")

print("\n" + "="*60)
print("DONE")
print("="*60)
