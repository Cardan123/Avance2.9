"""
Alternative model_manager.py using google.generativeai directly instead of langchain
This avoids the v1beta API issues
"""
import os
import json
from loguru import logger
import google.generativeai as genai
from dotenv import load_dotenv
from utils.config_file_manager import ConfigFileManager

class GeminiLLMWrapperDirect:
    """
    Direct wrapper for Google Gemini using google.generativeai SDK
    Avoids LangChain v1beta API compatibility issues
    """

    def __init__(self):
        logger.info(f"[GeminiLLMWrapperDirect] Initializing direct Gemini wrapper...")
        load_dotenv()
        
        self.retrieval_config_yaml_path = ConfigFileManager.default_yaml_path()
        self.retrieval_config = ConfigFileManager.load_yaml_config(self.retrieval_config_yaml_path)
        
        # Get model name - use simpler names without 'latest' suffix
        configured_model = self.retrieval_config.get("google_gemini_model_name", "gemini-1.5-pro")
        # Remove 'latest' suffix if present
        self.model_name = configured_model.replace("-latest", "")
        
        self.temperature = self.retrieval_config.get("google_gemini_temperature", 0.2)
        self.max_output_tokens = self.retrieval_config.get("google_gemini_max_output_tokens", 2048)

        # Get API key
        self.token = os.getenv("GOOGLE_API_KEY")
        if not self.token:
            raise ValueError("Google Gemini token is not set in environment variables.")

        # Configure genai
        genai.configure(api_key=self.token)
        
        # Create model
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
        )
        
        logger.info(f"[GeminiLLMWrapperDirect] Initialized with model: {self.model_name}")

    def invoke(self, messages):
        """
        Invoke the model with messages (compatible with LangChain interface)
        
        Args:
            messages: Can be a string, list of strings, or list of message dicts
        
        Returns:
            Response object with .content attribute
        """
        try:
            # Handle different message formats
            if isinstance(messages, str):
                # Simple string
                response = self.model.generate_content(messages)
            elif isinstance(messages, list):
                # List of messages
                if len(messages) == 1:
                    msg = messages[0]
                    
                    # Check if it's a LangChain message object
                    if hasattr(msg, 'content'):
                        content = msg.content
                        
                        # Handle multimodal content (text + image)
                        if isinstance(content, list):
                            parts = []
                            for part in content:
                                if isinstance(part, dict):
                                    if part.get("type") == "text":
                                        parts.append(part.get("text", ""))
                                    elif part.get("type") == "image_url":
                                        # Extract base64 image
                                        url = part.get("image_url", {}).get("url", "")
                                        if url.startswith("data:image"):
                                            # Parse data URL
                                            import base64
                                            from PIL import Image
                                            from io import BytesIO
                                            
                                            # Extract base64 part
                                            b64_data = url.split(",", 1)[1]
                                            image_data = base64.b64decode(b64_data)
                                            img = Image.open(BytesIO(image_data))
                                            parts.append(img)
                            
                            response = self.model.generate_content(parts)
                        else:
                            # Simple text content
                            response = self.model.generate_content(content)
                    else:
                        # Plain string in list
                        response = self.model.generate_content(msg)
                else:
                    # Multiple messages - join them
                    texts = []
                    for msg in messages:
                        if hasattr(msg, 'content'):
                            texts.append(str(msg.content))
                        else:
                            texts.append(str(msg))
                    response = self.model.generate_content("\n".join(texts))
            else:
                response = self.model.generate_content(str(messages))
            
            # Create response object compatible with LangChain
            class Response:
                def __init__(self, text, raw_response):
                    self.content = text
                    self.text = text
                    self.raw_response = raw_response
                    
                    # Extract additional metadata
                    self.additional_kwargs = {}
                    if hasattr(raw_response, 'candidates'):
                        self.additional_kwargs['candidates'] = [
                            {
                                'content': c.content,
                                'finishReason': c.finish_reason if hasattr(c, 'finish_reason') else None
                            }
                            for c in raw_response.candidates
                        ]
                    
                    if hasattr(raw_response, 'prompt_feedback'):
                        self.additional_kwargs['promptFeedback'] = raw_response.prompt_feedback
            
            return Response(response.text, response)
            
        except Exception as e:
            logger.error(f"[GeminiLLMWrapperDirect] Error invoking model: {e}")
            raise

    @property
    def llm(self):
        """Property for compatibility with existing code"""
        return self
