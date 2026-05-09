import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def gradcam_overlay_png_data_url(
    *,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_index: int,
    background_image: Image.Image,
    device: torch.device,
    alpha: float = 0.45,
) -> str:
    """
    Generate a Grad-CAM overlay and return it as a PNG data URL.
    Args:
        model:
            CNN model used for prediction.
        target_layer:
            Last convolutional layer to use for Grad-CAM.
        input_tensor:
            Preprocessed model input, shaped [1, 3, H, W].
            This should already be normalized the same way as training.
        target_index:
            Class index to explain.
        background_image:
            Normal-looking PIL image used for the overlay.
            This should NOT be normalized.
        alpha:
            Heatmap opacity. Higher = stronger heatmap.
    Returns:
        A string like:
            data:image/png;base64,...
    """

    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.to(device)

        # Clear old gradients.
        model.zero_grad(set_to_none=True)

        # Forward pass.
        output = model(input_tensor)

        # Some of your models may return (logits, features)
        # when return_features=True.
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape [B, C], got {logits.shape}")

        if target_index < 0 or target_index >= logits.shape[1]:
            raise ValueError(
                f"target_index={target_index} is out of range for logits shape {logits.shape}"
            )

        # Backward pass for the chosen class.
        target_score = logits[:, target_index].sum()
        target_score.backward()

        if activations is None:
            raise RuntimeError("Forward hook did not capture activations.")

        if gradients is None:
            raise RuntimeError("Backward hook did not capture gradients.")

        # Grad-CAM weights:
        # average gradient over spatial dimensions H and W.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps.
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Keep only positive influence.
        cam = F.relu(cam)

        # Resize CAM to match the background image.
        bg_width, bg_height = background_image.size
        cam = F.interpolate(
            cam,
            size=(bg_height, bg_width),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize CAM to [0, 1].
        cam = cam.squeeze().detach().cpu()
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.numpy()

        # Convert heatmap to RGB.
        # This creates a simple black → red → yellow heatmap.
        heatmap = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        heatmap[..., 0] = np.clip(255 * cam_np, 0, 255).astype(np.uint8)
        heatmap[..., 1] = np.clip(255 * np.maximum(cam_np - 0.5, 0) * 2, 0, 255).astype(
            np.uint8
        )
        heatmap[..., 2] = 0

        background = background_image.convert("RGB")
        background_np = np.array(background).astype(np.float32)

        heatmap_np = heatmap.astype(np.float32)

        overlay_np = (1 - alpha) * background_np + alpha * heatmap_np
        overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)

        overlay_image = Image.fromarray(overlay_np)

        # Convert PIL image to PNG data URL.
        buffer = BytesIO()
        overlay_image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    finally:
        forward_handle.remove()
        backward_handle.remove()
