"""
Verify gemini-2.0-flash-exp supports vision
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("="*60)
print("TESTING GEMINI-2.0-FLASH-EXP MULTIMODAL CAPABILITIES")
print("="*60)

model_name = "gemini-2.0-flash-exp"

# Test 1: Text only
print("\n" + "="*60)
print("Test 1: Text Only")
print("="*60)

try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Say 'Text works'")
    print(f"✅ Text response: {response.text}")
except Exception as e:
    print(f"❌ Text failed: {e}")
    exit(1)

# Test 2: Vision (image description)
print("\n" + "="*60)
print("Test 2: Vision (Image Description)")
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
    # Search for any image
    import os
    data_dir = r"C:\Git\MultimodalRAG\data"
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break

if not test_image:
    print("⚠️  No test image found - cannot verify vision support")
    print("But text works, so the model is functional!")
    exit(0)

print(f"Using image: {test_image}")

try:
    img = Image.open(test_image)
    print(f"Image size: {img.size}")
    
    # Resize if too large
    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
        print(f"Resized to: {img.size}")
    
    # Send to Gemini
    response = model.generate_content([
        "Describe this image in one sentence.",
        img
    ])
    
    print(f"\n✅✅ VISION WORKS!")
    print(f"\nImage description: {response.text}")
    
    print("\n" + "="*60)
    print("CONFIRMED: gemini-2.0-flash-exp supports:")
    print("  ✅ Text")
    print("  ✅ Vision (images)")
    print("  ✅ Multimodal (text + images together)")
    print("="*60)
    
except Exception as e:
    print(f"❌ Vision test failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Check if it's a specific error type
    error_str = str(e)
    if "not support" in error_str.lower() or "multimodal" in error_str.lower():
        print("\n⚠️  Model does NOT support vision!")
        print("Try another model like: gemini-2.0-flash or gemini-2.0-pro-exp")
    else:
        print("\n⚠️  Error may be due to image format or size")
