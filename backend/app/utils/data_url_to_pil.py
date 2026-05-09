import base64
from io import BytesIO

from PIL import Image


def data_url_to_pil(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image

