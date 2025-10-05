"""
Improved version of describe_single method with better error handling
Copy this into your ImageDescriptionChain class
"""

def describe_single_improved(self, image_path: str, extra_instruction: str = "") -> Dict[str, Any]:
    """Improved version with better debugging"""
    message, error = self._build_message(image_path, extra_instruction)
    if error:
        return error
    
    try:
        # Log message details
        logger.info(f"[ImageDescriptionChain] Processing: {image_path}")
        
        # Check if image content is properly formatted
        if isinstance(message.content, list):
            for i, part in enumerate(message.content):
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        logger.debug(f"  Part {i}: TEXT ({len(part.get('text', ''))} chars)")
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        logger.debug(f"  Part {i}: IMAGE (url length: {len(url)})")
        
        # IMPORTANT: Try different invocation methods
        try:
            # Method 1: Standard invoke with list
            response = self.llm.invoke([message])
            logger.debug(f"[ImageDescriptionChain] Response type: {type(response)}")
            logger.debug(f"[ImageDescriptionChain] Response: {response}")
        except Exception as e1:
            logger.warning(f"[ImageDescriptionChain] Method 1 failed: {e1}")
            try:
                # Method 2: Direct content invoke
                response = self.llm.invoke(message)
                logger.debug(f"[ImageDescriptionChain] Method 2 response: {response}")
            except Exception as e2:
                logger.error(f"[ImageDescriptionChain] Method 2 also failed: {e2}")
                return {
                    "path": image_path,
                    "error": f"Both invocation methods failed: {e1}, {e2}",
                    "empty_response": True
                }
        
        # Extract content
        raw_content = getattr(response, "content", "")
        
        # Check response metadata
        additional_kwargs = getattr(response, 'additional_kwargs', {})
        
        if additional_kwargs:
            logger.debug(f"[ImageDescriptionChain] additional_kwargs keys: {list(additional_kwargs.keys())}")
            
            # Check for safety blocks
            prompt_feedback = additional_kwargs.get("promptFeedback") or additional_kwargs.get("prompt_feedback")
            if prompt_feedback:
                logger.warning(f"[ImageDescriptionChain] Prompt feedback: {prompt_feedback}")
            
            # Check candidates
            candidates = additional_kwargs.get("candidates", [])
            if candidates:
                logger.debug(f"[ImageDescriptionChain] Found {len(candidates)} candidates")
                for idx, cand in enumerate(candidates):
                    finish_reason = cand.get("finishReason") or cand.get("finish_reason")
                    logger.debug(f"  Candidate {idx} finish_reason: {finish_reason}")
                    
                    # Try to extract text from candidates
                    if not raw_content:
                        content_obj = cand.get("content", {})
                        parts = content_obj.get("parts", [])
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                text = part.get("text", "")
                                if text:
                                    logger.info(f"[ImageDescriptionChain] Extracted text from candidate: {text[:100]}...")
                                    raw_content = text
                                    break
        
        # Process content
        if isinstance(raw_content, str):
            cleaned = raw_content.strip()
        elif isinstance(raw_content, list):
            texts = []
            for part in raw_content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif "text" in part:
                        texts.append(part["text"])
            cleaned = "\n".join(t.strip() for t in texts if t.strip())
        else:
            cleaned = str(raw_content)
        
        if not cleaned:
            logger.error(f"[ImageDescriptionChain] EMPTY RESPONSE for {image_path}")
            logger.error(f"  Raw content type: {type(raw_content)}")
            logger.error(f"  Raw content: {raw_content!r}")
            logger.error(f"  Additional kwargs: {additional_kwargs}")
            
            # Last resort: try without image
            logger.info("[ImageDescriptionChain] Retrying without image...")
            text_only = next((p.get("text") for p in message.content if isinstance(p, dict) and p.get("type") == "text"), None)
            if text_only:
                try:
                    retry_response = self.llm.invoke(text_only)
                    retry_content = getattr(retry_response, "content", "")
                    if retry_content:
                        logger.success(f"[ImageDescriptionChain] Text-only retry succeeded: {retry_content[:100]}...")
                        cleaned = retry_content
                except Exception as retry_err:
                    logger.error(f"[ImageDescriptionChain] Text-only retry failed: {retry_err}")
        
        # Try to parse JSON
        parsed_json = None
        if cleaned and "{" in cleaned and "}" in cleaned:
            try:
                start = cleaned.index("{")
                end = cleaned.rindex("}") + 1
                json_str = cleaned[start:end]
                parsed_json = json.loads(json_str)
                logger.success(f"[ImageDescriptionChain] Successfully parsed JSON")
            except Exception as json_err:
                logger.warning(f"[ImageDescriptionChain] Failed to parse JSON: {json_err}")
        
        return {
            "path": image_path,
            "filename": Path(image_path).name,
            "raw_description": cleaned,
            "parsed": parsed_json,
            "empty_response": not bool(cleaned),
            "finish_reasons": [c.get("finishReason") for c in additional_kwargs.get("candidates", [])],
            "safety_blocked": bool(additional_kwargs.get("promptFeedback")),
        }
        
    except Exception as e:
        logger.exception(f"[ImageDescriptionChain] Fatal error: {e}")
        return {
            "path": image_path,
            "error": str(e),
            "empty_response": True
        }
