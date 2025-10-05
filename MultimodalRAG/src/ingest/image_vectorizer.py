import torch
from PIL import Image

class ImageVectorizer:
    """
    Class for vectorizing images using multiple models (SAM, DINO).
    """

    def __init__(self, model_name: str = "dino", device: str = "cpu"):
        self.device = device
        self.model_name = model_name.lower()
        self.model = self._load_model(self.model_name)

    _dino_model = None

    def _load_model(self, model_name):
        if model_name == "dino":
            if ImageVectorizer._dino_model is None:
                from torchvision.models import dino_vits16
                model = dino_vits16(pretrained=True)
                model.eval()
                model.to(self.device)
                ImageVectorizer._dino_model = model
            return ImageVectorizer._dino_model
        else:
            raise ValueError(f"Model '{model_name}' not supported.")

    def vectorize(self, image_path: str) -> torch.Tensor:
        """
        Vectorizes an image using the selected model.

        Args:
            image_path (str): Path to the image.

        Returns:
            torch.Tensor: Image embedding vector.
        """
        image = Image.open(image_path).convert("RGB")
        # Apply preprocessing and inference according to the selected model
        if self.model_name == "sam":
            # Implement vectorization with SAM
            raise NotImplementedError("SAM vectorization not implemented yet.")
        elif self.model_name == "dino":
            # Implement vectorization with DINO
            raise NotImplementedError("DINO vectorization not implemented yet.")
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
        

""" if __name__ == "__main__":
    print("Testing DINO model loading...")
    try:
        vectorizer = ImageVectorizer(model_name="dino")
        if vectorizer.model:
            print("DINO model loaded successfully.")
        else:
            print("DINO model NOT loaded.")
    except Exception as e:
        print(f"Error loading DINO model: {e}") """