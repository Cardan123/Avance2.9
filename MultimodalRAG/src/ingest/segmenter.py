
import os
from PIL import Image, ImageDraw
from loguru import logger
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator

class Segmenter:
    """
    Class for automatic image segmentation (stub for SAM).
    """
    _sam_model = None
    _sam_predictor = None
    _sam_device = None

    def __init__(self, model_name="sam", checkpoint_path=None, device="cpu"):

        # Load SAM model only once
        self.model_name = model_name 
        if Segmenter._sam_model is None or Segmenter._sam_predictor is None or Segmenter._sam_device != device:
            if checkpoint_path is None:
                # Obtener el directorio base del proyecto
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                checkpoint_path = os.path.join(project_root, "src", "models", "SAM", "sam_vit_b_01ec64.pth")
            try:
                # Load SAM model
                logger.info(f"[Segmenter] Loading SAM model from {checkpoint_path} on device {device}...")
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
                sam.to(device)
                Segmenter._sam_model = sam
                Segmenter._sam_predictor = SamPredictor(sam)
                Segmenter._sam_device = device
            except ImportError:
                Segmenter._sam_model = None
                Segmenter._sam_predictor = None
                Segmenter._sam_device = None
                logger.info("[Segmenter] segment_anything not installed. SAM model not loaded.")
        else:
            logger.info("[Segmenter] SAM model already loaded, reusing it.")
            
        self.predictor = Segmenter._sam_predictor
        self.device = device
        logger.info(f"[Segmenter] Initialized with model: {self.model_name} on device: {self.device}")

    def segment(self, image_path, grid_size=16, mode="auto"):
        return self.segment_pipeline(image_path, grid_size, mode=mode)

    def segment_pipeline(self, image_path, grid_size=16, mode="auto"):
        """Modular pipeline for segmenting an image with SAM. Permite modo manual o automático."""
        # 1. Load and preprocess image
        image = self.load_image(image_path)
        logger.info(f"[Segmenter] Image loaded with size: {image.size}")
        #2. Preprocess image (if needed)
        image = self.preprocess_image(image)
        if self.predictor is None:
            logger.error("[Segmenter] SAM predictor not loaded. Returning whole image as single segment.")
            width, height = image.size
            return [{"image": image, "bbox": (0, 0, width, height)}]

        image_np = np.array(image)

        if mode == "manual":
            logger.info("[Segmenter] Running manual grid-based segmentation...")
            # 3. Prepare prompts (grid of points)
            points = self.prepare_prompts(image_np, grid_size)
            # 4. Run SAM predictor
            masks, scores, logits = self.run_sam_predictor(image_np, points)
            # 5. Extract and return segments
            segments = self.extract_segments(image, masks, scores=scores)
        elif mode == "auto":
            # 3. Run Automatic Mask Generator
            segments = self.run_sam_mask_generator(image, image_np)
        else:
            logger.error(f"[Segmenter] Unknown mode: {mode}. Returning whole image.")
            width, height = image.size
            return [{"image": image, "bbox": (0, 0, width, height)}]

        # If no segments found, return whole image
        if not segments:
            width, height = image.size
            return [{"image": image, "bbox": (0, 0, width, height)}]
        return segments
    
    def run_sam_mask_generator(self, image, image_np, score_threshold=0.7):
        """
        Ejecuta el Mask Generator y extrae los segmentos (wrapper para compatibilidad).
        """
        masks = self.get_auto_masks(image_np)
        return self.extract_auto_segments(image, masks, score_threshold=score_threshold)
    
    def get_auto_masks(self, image_np):
        """
        Get the automatic masks using SamAutomaticMaskGenerator.
        """
        logger.info("[Segmenter] Running SAM's automatic Mask Generator...")
        mask_generator = SamAutomaticMaskGenerator(self._sam_model)
        masks = mask_generator.generate(image_np)
        logger.info(f"[Segmenter] Mask Generator produced {len(masks)} masks.")
        return masks

    def extract_auto_segments(self, image, masks, score_threshold=0.7):
        """
        Extract segments from the automatic masks.      
        """
        logger.info(f"[Segmenter] Extracting segments from {len(masks)} automatic masks...")
        segments = []
        for m in masks:
            # Filter by score_threshold if present
            if "predicted_iou" in m and m["predicted_iou"] < score_threshold:
                continue
            seg_mask = m["segmentation"]
            ys, xs = np.where(seg_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bbox = (x1, y1, x2, y2)
            seg_img = image.crop(bbox)
            segments.append({
                "image": seg_img,
                "bbox": bbox,
                "score": m.get("predicted_iou", None),
                "area": m.get("area", None),
                "mask": seg_mask
            })
        logger.info(f"[Segmenter] Extracted {len(segments)} automatic segments (score > {score_threshold}).")
        return segments

    def load_image(self, image_path):
        """Load and convert the image to RGB."""
        logger.info(f"[Segmenter] Loading image from {image_path}...")
        return Image.open(image_path).convert("RGB")

    def preprocess_image(self, image):
        """
        Preprocess the image before segmentation.
        Placeholder: apply filters, normalization, etc.
        For now, return the image unchanged.
        """
        logger.info("[Segmenter] Preprocessing image / not implemented.")
        # return image.filter(...)
        return image
    
    def prepare_prompts(self, image_np, grid_size):
        """Generate prompt points in a grid."""
        logger.info(f"[Segmenter] Preparing {grid_size}x{grid_size} grid prompts...")
        h, w, _ = image_np.shape
        points = []
        for y in np.linspace(0, h-1, grid_size):
            for x in np.linspace(0, w-1, grid_size):
                points.append([int(x), int(y)])

        logger.info(f"[Segmenter] Generated {len(points)} prompt points.")
        return np.array(points)

    def run_sam_predictor(self, image_np, points):
        """Run the SAM predictor and get the masks."""
        logger.info(f"[Segmenter] Running SAM predictor with {len(points)} points...")
        self.predictor.set_image(image_np)
        input_labels = np.ones(len(points))
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=input_labels,
            multimask_output=True
        )

        logger.info(f"[Segmenter] SAM predictor returned {len(masks)} masks.")
        return masks, scores, logits

    def extract_segments(self, image, masks, scores=None, score_threshold=0.7):
        """Process the masks and extract the segments and bounding boxes, filtrando por score."""
        logger.info(f"[Segmenter] Extracting segments from {len(masks)} masks...")
        segments = []
        for i, mask in enumerate(masks):
            # Si hay scores, filtra por score_threshold
            if scores is not None and scores[i] < score_threshold:
                continue
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bbox = (x1, y1, x2, y2)
            seg_img = image.crop(bbox)
            segments.append({"image": seg_img, "bbox": bbox, "score": scores[i] if scores is not None else None})

        logger.info(f"[Segmenter] Extracted {len(segments)} segments (score > {score_threshold}).")
        return segments

    def list_and_show_segments(self, image_path, segments_info):
        """
        Lists and displays the found segments on the image.
        Args:
            image_path (str): Path to the original image.
            segments_info (list): List of dicts with segment info, e.g. bounding boxes.
        """
        logger.info(f"[Segmenter] Listing and showing {len(segments_info)} segments...")
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for idx, seg in enumerate(segments_info):
            # Example: seg = {"bbox": (x1, y1, x2, y2)}
            bbox = seg.get("bbox")
            if bbox:
                draw.rectangle(bbox, outline="red", width=2)
                draw.text((bbox[0], bbox[1]), f"{idx}", fill="yellow")
        image.show()
        # Optionally, return the image with drawn segments
        return image
    

    
""" if __name__ == "__main__":
    logger.info("Testing SAM model loading...")
    segmenter = Segmenter()
    if segmenter.predictor:
        logger.info("SAM model loaded successfully.")
    else:
        logger.error("SAM model NOT loaded.") """