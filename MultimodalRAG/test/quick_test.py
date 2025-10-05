"""
Quick test with different model names - no extra packages needed
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Models to try in order of likelihood
models_to_try = [
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-pro-vision",
    "gemini-1.0-pro-vision",
]

print("="*60)
print("QUICK MODEL TEST")
print("="*60)

for model_name in models_to_try:
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
        
        # Quick text test
        msg = HumanMessage(content="Reply with just 'OK'")
        response = llm.invoke([msg])
        content = getattr(response, "content", "")
        
        if content and len(content) > 0:
            print(f"  ✅ SUCCESS! Model '{model_name}' works!")
            print(f"  Response: {content}")
            print(f"\n  👉 Use this model name in config_retrieval.yaml")
            print(f"     google_gemini_model_name: {model_name}")
            break
        else:
            print(f"  ⚠️  Empty response")
            
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            print(f"  ❌ Model not available (404)")
        elif "403" in error_str:
            print(f"  ❌ Permission denied (403)")
        else:
            print(f"  ❌ Error: {error_str[:100]}")
            
print("\n" + "="*60)
print("If all failed, install google-generativeai:")
print("  pip install google-generativeai")
print("Then run list_models.py")
print("="*60)
