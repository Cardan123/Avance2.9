"""
Test available Gemini models
"""
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("="*60)
print("TESTING GEMINI MODELS")
print("="*60)

# List of models to test (compatible with v1beta API)
models_to_test = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-pro-vision",
    "gemini-1.5-flash-latest",
    "models/gemini-1.5-pro-latest",
]

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

for model_name in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_output_tokens=1024,
            api_key=api_key
        )
        print(f"  ✓ Model initialized")
        
        # Test text only
        msg = HumanMessage(content="Say 'Model OK'")
        response = llm.invoke([msg])
        content = getattr(response, "content", "")
        
        if content:
            print(f"  ✓ Response: '{content[:50]}...'")
            print(f"\n  ✅ SUCCESS: {model_name} works!")
            
            # If this model works, test with image
            print(f"\n  Testing with image...")
            
            import base64
            from pathlib import Path
            from PIL import Image
            from io import BytesIO
            
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
            
            if test_image:
                with Image.open(test_image) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    if max(w, h) > 1024:
                        scale = 1024 / max(w, h)
                        img = img.resize((int(w * scale), int(h * scale)))
                    
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=90)
                    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                img_msg = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                )
                
                img_response = llm.invoke([img_msg])
                img_content = getattr(img_response, "content", "")
                
                if img_content:
                    print(f"  ✓ Image response: '{img_content[:100]}...'")
                    print(f"\n  ✅✅ PERFECT: {model_name} supports vision!")
                else:
                    print(f"  ⚠ Image response was empty")
            else:
                print(f"  ⚠ No test image found")
                
        else:
            print(f"  ⚠ Empty response")
            
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"  ✗ Model not found (404)")
        elif "400" in error_msg:
            print(f"  ✗ Bad request (400)")
        else:
            print(f"  ✗ Error: {error_msg[:100]}")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("\nBased on the tests above, use the first model that shows:")
print("  ✅✅ PERFECT: [model] supports vision!")
print("\nUpdate config_retrieval.yaml with that model name.")
