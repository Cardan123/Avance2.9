"""
List available Gemini models from Google API
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found!")
    exit(1)

print("="*60)
print("LISTING AVAILABLE GEMINI MODELS")
print("="*60)

genai.configure(api_key=api_key)

print("\nFetching models from Google API...")
try:
    models = genai.list_models()
    
    print("\n" + "="*60)
    print("AVAILABLE MODELS:")
    print("="*60)
    
    vision_models = []
    text_models = []
    
    for model in models:
        name = model.name
        supported_methods = model.supported_generation_methods
        
        # Check if it supports generateContent (required for chat)
        if 'generateContent' in supported_methods:
            # Check if it supports vision
            supports_vision = any(method in ['generateContent'] for method in supported_methods)
            
            print(f"\n📦 {name}")
            print(f"   Supported methods: {', '.join(supported_methods)}")
            
            # Try to detect vision support from name
            if 'vision' in name.lower() or '1.5' in name or '2.0' in name:
                vision_models.append(name)
                print(f"   👁️  Likely supports vision")
            else:
                text_models.append(name)
                print(f"   📝 Text only")
    
    print("\n" + "="*60)
    print("RECOMMENDED MODELS FOR MULTIMODAL RAG:")
    print("="*60)
    
    if vision_models:
        print("\n✅ Models with vision support:")
        for model in vision_models:
            # Extract just the model name without 'models/' prefix
            model_name = model.replace('models/', '')
            print(f"   - {model_name}")
        
        print("\n💡 Use one of these in your config_retrieval.yaml:")
        print(f"   google_gemini_model_name: {vision_models[0].replace('models/', '')}")
    else:
        print("\n⚠️ No vision models found")
    
    if text_models:
        print("\n📝 Text-only models (NOT suitable for images):")
        for model in text_models[:3]:
            model_name = model.replace('models/', '')
            print(f"   - {model_name}")

except Exception as e:
    print(f"\n❌ Error listing models: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Try direct API call
    try:
        import requests
        url = "https://generativelanguage.googleapis.com/v1/models"
        params = {"key": api_key}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*60)
            print("MODELS FROM DIRECT API:")
            print("="*60)
            
            for model in data.get("models", []):
                name = model.get("name", "")
                print(f"\n📦 {name}")
                if "supportedGenerationMethods" in model:
                    methods = model["supportedGenerationMethods"]
                    print(f"   Methods: {', '.join(methods)}")
        else:
            print(f"API returned status {response.status_code}")
            print(response.text)
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")

print("\n" + "="*60)
print("TESTING DIRECT GENAI SDK:")
print("="*60)

# Test with google.generativeai directly (not langchain)
try:
    # Try common model names
    test_models = [
        "gemini-1.5-pro",
        "gemini-1.5-flash", 
        "gemini-pro-vision",
        "gemini-1.0-pro-vision",
    ]
    
    for model_name in test_models:
        try:
            print(f"\nTesting: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Test text
            response = model.generate_content("Say 'OK'")
            print(f"  ✓ Text response: {response.text[:50]}")
            
            # Test with image
            from PIL import Image
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
                img = Image.open(test_image)
                response = model.generate_content(["Describe briefly", img])
                print(f"  ✓ Vision response: {response.text[:100]}")
                print(f"  ✅✅ SUCCESS: {model_name} works with vision!")
                
        except Exception as e:
            error_str = str(e)
            if "not found" in error_str.lower() or "404" in error_str:
                print(f"  ✗ Model not found")
            else:
                print(f"  ✗ Error: {error_str[:100]}")
                
except Exception as e:
    print(f"SDK test failed: {e}")

print("\n" + "="*60)
print("DONE")
print("="*60)
