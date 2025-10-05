from __future__ import annotations
import base64
from pathlib import Path
from io import BytesIO
from typing import Optional
from PIL import Image
from loguru import logger


def image_file_to_base64(path: str, force_rgb: bool = True, max_side: Optional[int] = 1600) -> tuple[str, str]:
    """Load an image from disk and return (mime_subtype, base64_string).

    Args:
        path: Path to the image file.
        force_rgb: Convert image to RGB (avoids issues with PNG transparency for some models).
        max_side: If provided, downscale the image keeping aspect ratio so the largest side == max_side.

    Returns:
        (format_lowercase, base64_encoded_data_url_fragment)
    """
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(img_path) as img:
            original_format = (img.format or "JPEG").lower()
            if force_rgb:
                img = img.convert("RGB")
                # If original was png keep jpeg output to reduce size.
                target_format = "JPEG"
            else:
                target_format = img.format or "JPEG"

            if max_side:
                w, h = img.size
                largest = max(w, h)
                if largest > max_side:
                    scale = max_side / largest
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size)

            buffer = BytesIO()
            img.save(buffer, format=target_format, optimize=True, quality=90)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mime_subtype = (target_format or "jpeg").lower()
            # Normalize jpeg subtype name
            if mime_subtype == "jpg":
                mime_subtype = "jpeg"
            return mime_subtype, b64
    except Exception as e:
        logger.exception(f"[image_utils] Error converting image to base64: {e}")
        raise


def build_data_url(mime_subtype: str, b64: str) -> str:
    return f"data:image/{mime_subtype};base64,{b64}"
