import os
import uuid
from datetime import datetime
from PIL import Image

from config import OUTPUT_DIR


def resize_to_portrait(img: Image.Image, target_size=(832, 1216)) -> Image.Image:
    target_w, target_h = target_size
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))

    return img.resize(target_size, Image.LANCZOS)


def save_image(image: Image.Image, job_id: str = None) -> tuple[str, str]:
    """Save image to OUTPUT_DIR and return (filename, filepath)."""
    filename = f"{job_id}.png" if job_id else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    return filename, filepath
