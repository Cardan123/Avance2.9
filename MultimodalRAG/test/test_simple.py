"""
Simple diagnostic script - run from project root with: python -m test_simple
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Test 1: Check environment
print("="*60)
print("ENVIRONMENT CHECK")
print("="*60)

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"✓ GOOGLE_API_KEY found: {api_key[:15]}...")
else:
    print("✗ GOOGLE_API_KEY not found!")
    exit(1)

# Test 2: Import test
print("\n" + "="*60)
print("IMPORT CHECK")
print("="*60)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✓ langchain_google_genai imported")
except ImportError as e:
    print(f"✗ Failed to import langchain_google_genai: {e}")
    exit(1)

try:
    from langchain_core.messages import HumanMessage
    print("✓ langchain_core imported")
except ImportError as e:
    print(f"✗ Failed to import langchain_core: {e}")
    exit(1)

# Test 3: Initialize LLM
print("\n" + "="*60)
print("LLM INITIALIZATION TEST")
print("="*60)

models_to_test = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash"
]

for model_name in models_to_test:
    print(f"\nTesting model: {model_name}")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_output_tokens=1024,
            api_key=api_key
        )
        print(f"  ✓ Model initialized")
        
        # Test text-only
        msg = HumanMessage(content="Say 'OK'")
        response = llm.invoke([msg])
        content = getattr(response, "content", "")
        print(f"  ✓ Text response: '{content}'")
        
        if not content:
            print(f"  ✗ WARNING: Empty response!")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Test 4: Test with image
print("\n" + "="*60)
print("IMAGE TEST")
print("="*60)

import base64
from pathlib import Path

# Find an image
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
    print("✗ No test image found")
    print("\nSearching for images...")
    data_dir = r"C:\Git\MultimodalRAG\data"
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    print(f"  Found: {os.path.join(root, file)}")
                    if not test_image:
                        test_image = os.path.join(root, file)
    
    if not test_image:
        print("\n✗ No images found. Cannot test image processing.")
        exit(0)

print(f"\nUsing test image: {test_image}")

try:
    from PIL import Image
    with Image.open(test_image) as img:
        # Convert to RGB and resize
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 1024:
            scale = 1024 / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size)
        
        # Convert to base64
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        print(f"✓ Image loaded and converted (size: {len(b64)} chars)")
        
        # Test with best model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens=1024,
            api_key=api_key
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        )
        
        print("\nSending request to Gemini...")
        response = llm.invoke([message])
        
        content = getattr(response, "content", "")
        additional_kwargs = getattr(response, "additional_kwargs", {})
        
        print(f"\nResponse type: {type(response)}")
        print(f"Content: '{content}'")
        print(f"Content length: {len(content)}")
        
        if additional_kwargs:
            print(f"\nAdditional kwargs keys: {list(additional_kwargs.keys())}")
            
            # Check for blocks
            candidates = additional_kwargs.get("candidates", [])
            if candidates:
                for i, cand in enumerate(candidates):
                    finish_reason = cand.get("finishReason", "N/A")
                    print(f"  Candidate {i} finish_reason: {finish_reason}")
        
        if content:
            print("\n✓ SUCCESS: Image description received!")
        else:
            print("\n✗ WARNING: Empty response from image description")
            
except Exception as e:
    print(f"\n✗ Image test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
