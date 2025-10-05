from loguru import logger

class Classifier:
    """
    Classifier for assigning labels to image segments.
    For now, supports manual labeling. Can be extended for automatic classification.
    """
    def __init__(self):
        pass

    def classify(self, segment_img, manual_label=None):
        """
        Assigns a label to the segment. If manual_label is provided, uses it.
        In the future, this method can be extended to use a model for automatic classification.

        Args:
            segment_img: The image segment (PIL Image or ndarray).
            manual_label (str, optional): The label to assign manually.

        Returns:
            str: The assigned label.
        """
        logger.info("[Classifier] Classifying segment...")
        if manual_label:
            logger.info(f"[Classifier] Using manual label: {manual_label}")
            return manual_label
        # Placeholder for future automatic classification

        logger.info("[Classifier] No manual label provided. Assigning default label 'unlabeled'.")
        return "unlabeled"
