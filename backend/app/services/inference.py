import torch
import torch.nn as nn
from PIL import Image

from app.models.feature_extraction import extract_features
from app.utils.gradcam import gradcam_overlay_png_data_url
from app.utils.image_preprocessing import transform


class InferenceService:
    def __init__(
        self,
        *,
        shufflenet_model: torch.nn.Module,
        efficientnet_model: torch.nn.Module,
        resnet_model: torch.nn.Module,
        fusion_model: torch.nn.Module,
        prediction_block: torch.nn.Module,
        device: torch.device,
        class_names: list[str],
    ):
        self.fusion_model = fusion_model
        self.models = {
            "resnet": resnet_model,
            "efficientnet": efficientnet_model,
            "shufflenet": shufflenet_model,
        }
        self.prediction_block = prediction_block
        self.device = device
        self.class_names = class_names

        # Layers used for Grad-CAM (conv feature maps, before pooling).
        self._cam_layers = {
            "resnet": self.models["resnet"].backbone.resnet[-2],
            "efficientnet": self.models["efficientnet"].features[-1],
            "shufflenet": self.models["shufflenet"].backbone.conv5,
        }

    def predict_from_pil(self, image: Image.Image, *, top_k: int = 5):
        image_rgb = image.convert("RGB")
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        background_image = image_rgb.resize((224, 224))

        with torch.no_grad():
            f_shuffle = extract_features(
                self.models["shufflenet"], image_tensor, self.device
            )
            f_eff = extract_features(self.models["efficientnet"], image_tensor, self.device)
            f_res = extract_features(self.models["resnet"], image_tensor, self.device)

            fused, weights, _, _ = self.fusion_model(f_shuffle, f_eff, f_res)
            logits = self.prediction_block(fused)
            probs = torch.sigmoid(logits).squeeze(0)

            top_k = max(1, min(int(top_k), probs.numel()))
            values, indices = torch.topk(probs, k=top_k)

            fused_top1_index = int(indices[0].item()) if indices.numel() > 0 else None

            results = []
            for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
                label = (
                    self.class_names[idx] if idx < len(self.class_names) else str(idx)
                )
                results.append(
                    {"label": label, "probability": float(score), "index": int(idx)}
                )

        cams = self._gradcams_for_target(
            image_tensor=image_tensor,
            background_image=background_image,
            target_index=fused_top1_index,
        )

        return {
            "top_k": results,
            "fusion_weights": weights.squeeze(0).detach().cpu().tolist(),
            "explanations": {
                "target_index": fused_top1_index,
                "target_label": (
                    self.class_names[fused_top1_index]
                    if fused_top1_index is not None
                    and fused_top1_index < len(self.class_names)
                    else None
                ),
                "cams": cams,
            },
        }

    def _gradcams_for_target(
        self,
        *,
        image_tensor: torch.Tensor,
        background_image: Image.Image,
        target_index: int | None,
    ):
        if target_index is None:
            return {}

        cams: dict[str, dict[str, str]] = {}
        with torch.enable_grad():
            for model_name, model in self.models.items():
                cams[model_name] = {
                    "overlay_png": gradcam_overlay_png_data_url(
                    model=model,
                    target_layer=self._cam_layers[model_name],
                    input_tensor=image_tensor,
                    target_index=target_index,
                    background_image=background_image,
                    device=self.device,
                    )
                }
        return cams
