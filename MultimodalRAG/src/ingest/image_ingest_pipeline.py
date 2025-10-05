import os
import yaml
from transformers import CLIPProcessor, CLIPModel
#from ingest import segmenter
from ingest.segmenter import Segmenter
from ingest.image_vectorizer import ImageVectorizer
from ingest.mongo_client import MongoClient
from ingest.document_factory import DocumentFactory
from ingest.classifier import Classifier
from loguru import logger
import json
from PIL import Image
import pytesseract

class ImageIngestPipeline:
    """
    Pipeline to segment images in a folder, vectorize each segment, and store embeddings in MongoDB.
    """
    def __init__(self, config_path=None, image_folder=None):
        # Si no se proporciona config_path, usar la ruta absoluta por defecto
        if config_path is None:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_root, "config_ingest.yaml")
        
        self.config = self._load_config(config_path)
        self.sam_mode = self.config.get('sam_mode', 'auto')
        self.segmenter = Segmenter(model_name="sam")
        self.embedding_clip_model, self.embedding_clip_processor = self._load_clip_model()
        self.mongo = MongoClient()
        self.image_folder = self._check_image_folder(image_folder, config_path)
        self.recreate_vector_db = str(self.config.get('RECREATE_VECTOR_DB', 'False')).lower() in ("true", "1", "t")

        # Configurar la ruta de Tesseract desde self.config
        tesseract_path = self.config.get('tesseract_path')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"[ImageIngestPipeline] Tesseract path set to: {tesseract_path}")
        else:
            logger.warning("[ImageIngestPipeline] Tesseract path not found in config file.")

    def run(self):
        logger.info("[ImageIngestPipeline] Starting image ingestion pipeline...")
        if self.recreate_vector_db:
            logger.info("[ImageIngestPipeline] recreate_vector_db is set to True. Recreating the vector database...")
            self.process_all_images()
        else:
            logger.info("[ImageIngestPipeline] recreate_vector_db is set to False. Skipping database recreation.")
        logger.info("[ImageIngestPipeline] Image ingestion pipeline completed.")

    def _load_config(self, config_path):
        """
        Loads the YAML config file and returns the config dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[ImageIngestPipeline] Loaded config from {config_path}")

            return config
        except Exception as e:
            logger.error(f"[ImageIngestPipeline] Error loading config from {config_path}: {e}")
            return {}

    def process_all_images(self, pre_segmented_dict=None):
        """
        Processes all images in the folder.
        If pre_segmented_dict is provided, uses pre-segmented images and labels for those images.
        Args:
            pre_segmented_dict (dict, optional): Dictionary where keys are image paths and values are lists of segments (dicts with 'image', 'label', 'bbox').

        Example usage for pre-segmented images:
            pre_segmented_dict = {
                "path/to/image1.png": [
                    {"image": img1, "label": "symbol", "bbox": (x1, y1, x2, y2)},
                    {"image": img2, "label": "text", "bbox": (x3, y3, x4, y4)}
                ],
                "path/to/image2.png": [ ... ]
            }
            pipeline.process_all_images(pre_segmented_dict=pre_segmented_dict)
        """
        if not self.image_folder or not os.path.isdir(self.image_folder):
            raise ValueError("Image folder not found.")
        
        if pre_segmented_dict is None:
            pre_segmented_dict = self.load_presegmented_dict_from_json()
    
        for filename in os.listdir(self.image_folder):
            logger.info(f"[ImageIngestPipeline] Processing file in extension required: {filename}...")
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(self.image_folder, filename)
                if pre_segmented_dict and image_path in pre_segmented_dict:
                    self.process_image(image_path, pre_segmented=pre_segmented_dict[image_path])
                else:
                    self.process_image(image_path)


    def process_image(self, image_path, manual_labels=None, pre_segmented=None, save_segments=True, output_dir=None):
        """
        Process an image: segment, classify, vectorize, perform OCR, and store in MongoDB.
        """
        # Si no se proporciona output_dir, usar ruta absoluta por defecto
        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(project_root, "data", "images", "segments")
        segments = self.get_segments(image_path, pre_segmented)
        if save_segments:
            segments = self.save_segments_to_folder(image_path, segments, output_dir) 
        self.visualize_segments(image_path, segments, pre_segmented)

        logger.info(f"[ImageIngestPipeline] Processing {len(segments)} segments for image {image_path}...") 
        documentList = []
        for idx, seg in enumerate(segments):
            label = self.get_label(seg, idx, manual_labels, pre_segmented)
            embedding = self.get_clip_embedding(seg["image"])
            
            # Perform OCR on the segment
            text, ocr_bboxes = self.extract_text(seg["image"])
            
            # Create document with OCR data
            doc = self.create_document(
                image_id=image_path,
                embedding=embedding,
                image_path=image_path,
                segment_path=seg.get("seg_path"),
                label=label,
                bbox=seg.get("bbox"),
                text=text,
                ocr_bboxes=ocr_bboxes  
            )
            documentList.append(doc)
            # Uncomment to insert into MongoDB
            # self.mongo.insert_embedding(doc)

    def save_segments_to_folder(self, image_path, segments, output_dir):
        """
        Save each segment into a folder and update segments with their saved paths.
        """
        logger.info(f"[ImageIngestPipeline] Saving segments for {image_path} to folder {output_dir}...")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        folder_path = os.path.join(output_dir, base_name)
        os.makedirs(folder_path, exist_ok=True)
        for idx, seg in enumerate(segments):
            seg_img = seg["image"]
            seg_path = os.path.join(folder_path, f"segment_{idx}.png")
            seg_img.save(seg_path)
            seg["seg_path"] = seg_path  # Update segment with its saved path
            logger.info(f"[ImageIngestPipeline] Saved segment {idx} to {seg_path}")
        return segments

    def get_segments(self, image_path, pre_segmented):
        """Get segments: use pre_segmented if available, otherwise segment automatically."""
        if pre_segmented:
            return pre_segmented
        segments = self.segmenter.segment(image_path, mode=self.sam_mode)
        return segments

    def visualize_segments(self, image_path, segments, pre_segmented):
        """Show segments on the image if not pre-segmented."""
        logger.info(f"[ImageIngestPipeline] Visualizing segments for {image_path}...")
        if not pre_segmented:
            segments_info = [{"bbox": seg["bbox"]} for seg in segments]
            self.segmenter.list_and_show_segments(image_path, segments_info)

    def get_label(self, seg, idx, manual_labels, pre_segmented):
        """Get the label of the segment: from pre_segmented, manual_labels, or classifier."""
        logger.info(f"[ImageIngestPipeline] Getting label for segment {idx}...")
        if pre_segmented:
            logger.info(f"[ImageIngestPipeline] Using pre-segmented label: {seg.get('label', 'unlabeled')}")
            return seg.get("label", "unlabeled")
        
        logger.info(f"[ImageIngestPipeline] No pre-segmented label. Checking manual labels...")
        manual_label = None
        if manual_labels and idx < len(manual_labels):
            manual_label = manual_labels[idx]
        classifier = Classifier()
        return classifier.classify(seg["image"], manual_label=manual_label)

    def get_clip_embedding(self, segment_img):
        """Get the CLIP embedding for the segment."""
        logger.info(f"[ImageIngestPipeline] Generating CLIP embedding for segment...")
        inputs = self.embedding_clip_processor(images=segment_img, return_tensors="pt")
        embedding = self.embedding_clip_model.get_image_features(**inputs)

        logger.info(f"[ImageIngestPipeline] Generated embedding of shape {embedding.shape}.")
        return embedding

    def create_document(self, image_id, 
                        embedding, 
                        image_path, 
                        segment_path, 
                        label, 
                        bbox, 
                        text, 
                        ocr_bboxes):
        """Create the document for MongoDB with the relevant data."""
        logger.info(f"[ImageIngestPipeline] Creating document for MongoDB using segment_document...")
        doc = DocumentFactory.segment_document(
            image_id=image_id,
            image_embedding=embedding, 
            image_path=image_path, 
            segment_path=segment_path,
            label=label, 
            bbox=bbox, 
            text=text, 
            text_embedding=None,
            ocr_bboxes=ocr_bboxes
        )
        return doc
    
    def _check_image_folder(self, image_folder, config_path):
        """
        Verifies and assigns the image_folder path from parameter or config.
        """
        folder = image_folder
        if config_path and not image_folder:
            logger.info(f"[ImageIngestPipeline] Loading configuration from {config_path}...")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                folder = config.get('image_folder')
        if not folder:
            logger.error("[ImageIngestPipeline] Image folder not specified in config or as a parameter.")
            raise ValueError("Image folder must be specified either in config or as a parameter.")
        else:
            logger.info(f"[ImageIngestPipeline] Using image folder: {folder}")
        return folder

    def _load_clip_model(self):
        """
        Search for the CLIP model in the local cache (snapshots) or download it if it doesn't exist.
        """
        from transformers import CLIPModel, CLIPProcessor
        from utils.model_downloader import CLIPModelDownloader
        token = os.getenv("HUGGINGFACE_TOKEN")
        downloader = CLIPModelDownloader()
        snapshot_path = downloader.get_snapshot_path()
        if snapshot_path:
            logger.info(f"[ImageIngestPipeline] Loading CLIP model from cache: {snapshot_path}")
            model = CLIPModel.from_pretrained(snapshot_path)
            processor = CLIPProcessor.from_pretrained(snapshot_path)
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            local_embedding_model_path = os.path.join(project_root, "src", "models", "CLIP", "openai_clip-vit-base-patch16")
            logger.info(f"[ImageIngestPipeline] Downloading CLIP model to {local_embedding_model_path}...")
            model, processor = downloader.get_model(use_auth_token=token)
        return model, processor
    

    def load_presegmented_dict_from_json(self, config_path=None, image_lookup=None):
        """
        Read the segment JSON and build the pre_segmented_dict for the pipeline.
        Args:
            config_path (str): Path to the YAML config file.
            image_lookup (dict, optional): Dictionary to map id to PIL image, if needed.
        Returns:
            dict: {image_path: [segment_dict, ...], ...}
        """
        # Si no se proporciona config_path, usar la ruta absoluta por defecto
        if config_path is None:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_root, "config_ingest.yaml")
            
        # Get the JSON path from config
        json_path = self.get_segmented_images_dataset_path(config_path)
        # Get the base folder of the images (one level above the JSON)
        base_folder = os.path.dirname(os.path.dirname(json_path))
        # Read the JSON, check if it exists
        try:
            if not os.path.isfile(json_path):
                logger.warning(f"[ImageIngestPipeline] The segment json file does not exist: {json_path}")
                return None
            with open(json_path, "r") as f:
                segments_list = json.load(f)
        except Exception as e:
            logger.error(f"[ImageIngestPipeline] Error opening the segment json file: {json_path}: {e}")
            return None
        # Build the dictionary
        result = {}
        logger.info(f"[ImageIngestPipeline] Building pre-segmented dictionary from {json_path}...")
        for obj in segments_list:
            image_name = obj["id"]
            image_path = os.path.join(base_folder, image_name)
            logger.info(f"[ImageIngestPipeline] Processing segment for image: {image_path}")
            bbox = self.manual_bbox_to_pipeline_bbox(obj["boundingBox"])
            label = obj["tags"][0] if obj["tags"] else "unlabeled"
            # Load the image and apply the crop
            try:
                logger.info(f"[ImageIngestPipeline] Loading and cropping image for segment: {image_path} with bbox {bbox}")
                img = Image.open(image_path)
                seg_img = img.crop(bbox)
                logger.info(f"[ImageIngestPipeline] Cropped segment image size: {seg_img.size}")
            except Exception as e:
                logger.error(f"[ImageIngestPipeline] Error loading or cropping image in segmented dictionary {image_path}: {e}")
                seg_img = None
            segment_dict = {"image": seg_img, "label": label, "bbox": bbox}
            if image_path not in result:
                result[image_path] = []
            result[image_path].append(segment_dict)
        logger.info(f"[ImageIngestPipeline] Built pre-segmented dictionary with {len(result)} images.")
        return result
    
    def get_segmented_images_dataset_path(self, config_path=None):
        if config_path is None:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_root, "config_ingest.yaml")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("segmented_images_dataset")
    
    def manual_bbox_to_pipeline_bbox(self, manual_bbox):
        """
        Convierte un bounding box de VoTT al formato (x1, y1, x2, y2) para el pipeline.
        Args:
            vott_bbox (dict): Diccionario con las claves 'left', 'top', 'width', 'height'.
        Returns:
            tuple: (x1, y1, x2, y2)
        """
        x1 = manual_bbox["left"]
        y1 = manual_bbox["top"]
        x2 = x1 + manual_bbox["width"]
        y2 = y1 + manual_bbox["height"]
        return (x1, y1, x2, y2)

    def extract_text(self, segment_img):
        """
        Extract text from a segment image using OCR.
        """
        logger.info("[ImageIngestPipeline] Extracting text from segment using OCR...")
        try:
            # Convert image to grayscale for better OCR results
            gray_img = segment_img.convert("L")
            # Perform OCR
            ocr_result = pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT)
            # Extract text and bounding boxes
            extracted_text = " ".join([ocr_result['text'][i] for i in range(len(ocr_result['text'])) if ocr_result['text'][i].strip()])
            bounding_boxes = [
                {
                    "bbox": (ocr_result['left'][i], ocr_result['top'][i],
                              ocr_result['left'][i] + ocr_result['width'][i],
                              ocr_result['top'][i] + ocr_result['height'][i]),
                    "text": ocr_result['text'][i]
                }
                for i in range(len(ocr_result['text'])) if ocr_result['text'][i].strip()
            ]
            logger.info(f"[ImageIngestPipeline] Extracted text: {extracted_text}")
            return extracted_text, bounding_boxes
        except Exception as e:
            logger.error(f"[ImageIngestPipeline] Error during OCR: {e}")
            return None, []
