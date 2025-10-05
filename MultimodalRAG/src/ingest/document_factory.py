from loguru import logger
class DocumentFactory:
    """
    Factory class to create MongoDB documents for image embeddings.
    """
    @staticmethod
    def default_document(model_name, embedding, image_path):
        logger.info(f"[DocumentFactory] Creating document for model: {model_name}")

        doc = {
            "model": model_name,
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            "source_image": image_path,
        }
        logger.info(f"[DocumentFactory] Created document.")
        
        return doc

    # We can add more static methods here for custom document formats
    # Example:
    # @staticmethod
    # def custom_document(...):
    #     return {...}

    @staticmethod
    def segment_document(
        image_id,
        image_path,
        segment_path,
        image_embedding,
        text,
        text_embedding,
        label,
        bbox,
        ocr_bboxes,
        metadata=None
    ):
        """
        Creates a MongoDB document for a segmented image with both image and text embeddings.
        """
        logger.info(f"[DocumentFactory] Creating segment document for image_id: {image_id}")

        doc = {
            "image_id": image_id,
            "image_path": image_path,
            "segment_path": segment_path,
            "image_embedding": image_embedding.tolist() if hasattr(image_embedding, "tolist") else image_embedding,
            "text": text,
            "text_embedding": text_embedding.tolist() if hasattr(text_embedding, "tolist") else text_embedding,
            "label": label,
            "bbox": bbox,
            "metadata": metadata or {}
        }
        logger.info(f"[DocumentFactory] Created segment document.")
        return doc
