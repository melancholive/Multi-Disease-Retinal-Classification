import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2


def crop_black_borders(image: Image.Image) -> Image.Image:
    img_np = np.array(image)

    if img_np.ndim == 2:
        return image

    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    black_threshold = 10
    non_black_pixels = np.any(img_np > black_threshold, axis=2)

    if not np.any(non_black_pixels):
        return image

    rows = np.any(non_black_pixels, axis=1)
    cols = np.any(non_black_pixels, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_img_np = img_np[rmin : rmax + 1, cmin : cmax + 1]
    return Image.fromarray(cropped_img_np)


transform = v2.Compose(
    [
        v2.Lambda(crop_black_borders),
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
