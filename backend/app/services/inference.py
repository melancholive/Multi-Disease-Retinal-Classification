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
        self.shufflenet_model = shufflenet_model
        self.efficientnet_model = efficientnet_model
        self.resnet_model = resnet_model
        self.fusion_model = fusion_model
        self.prediction_block = prediction_block
        self.device = device
        self.class_names = class_names

        # Layers used for Grad-CAM (conv feature maps, before pooling).
        self._cam_layers = {
            "resnet": self._find_resnet_cam_layer(self.resnet_model),
            "efficientnet": self._find_efficientnet_cam_layer(self.efficientnet_model),
            "shufflenet": self._find_shufflenet_cam_layer(self.shufflenet_model),
        }

    @staticmethod
    def _find_resnet_cam_layer(model: nn.Module) -> nn.Module:
        backbone = getattr(model, "backbone", None)
        resnet = getattr(backbone, "resnet", None)
        # Some backbones keep the original torchvision ResNet module with `layer4`.
        layer4 = getattr(resnet, "layer4", None)
        if layer4 is not None:
            return layer4

        # In this repo, `ResNetBackbone` wraps children into an `nn.Sequential`,
        # which removes attribute names. That sequential is typically:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool].
        if isinstance(resnet, nn.Sequential) and len(resnet) >= 2:
            return resnet[-2]

        raise AttributeError(
            "Could not locate ResNet CAM layer (backbone.resnet.layer4 or sequential[-2])"
        )

    @staticmethod
    def _find_efficientnet_cam_layer(model: nn.Module) -> nn.Module:
        features = getattr(model, "features", None)
        if features is None:
            raise AttributeError("Could not locate EfficientNet CAM layer (features)")
        return features[-1]

    @staticmethod
    def _find_shufflenet_cam_layer(model: nn.Module) -> nn.Module:
        backbone = getattr(model, "backbone", None)
        # Prefer the last conv stage if present (torchvision shufflenet has conv5).
        conv5 = getattr(backbone, "conv5", None)
        if conv5 is not None:
            return conv5

        # Fallback: last child module.
        children = list(backbone.children()) if backbone is not None else []
        if not children:
            raise AttributeError("Could not locate ShuffleNet CAM layer")
        return children[-1]

    def predict_from_pil(self, image: Image.Image, *, top_k: int = 5):
        image_rgb = image.convert("RGB")
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        background_image = image_rgb.resize((224, 224))

        with torch.no_grad():
            f_shuffle = extract_features(
                self.shufflenet_model, image_tensor, self.device
            )
            f_eff = extract_features(self.efficientnet_model, image_tensor, self.device)
            f_res = extract_features(self.resnet_model, image_tensor, self.device)

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
            cams["resnet"] = {
                "overlay_png": gradcam_overlay_png_data_url(
                    model=self.resnet_model,
                    target_layer=self._cam_layers["resnet"],
                    input_tensor=image_tensor,
                    target_index=target_index,
                    background_image=background_image,
                    device=self.device,
                )
            }
            cams["efficientnet"] = {
                "overlay_png": gradcam_overlay_png_data_url(
                    model=self.efficientnet_model,
                    target_layer=self._cam_layers["efficientnet"],
                    input_tensor=image_tensor,
                    target_index=target_index,
                    background_image=background_image,
                    device=self.device,
                )
            }
            cams["shufflenet"] = {
                "overlay_png": gradcam_overlay_png_data_url(
                    model=self.shufflenet_model,
                    target_layer=self._cam_layers["shufflenet"],
                    input_tensor=image_tensor,
                    target_index=target_index,
                    background_image=background_image,
                    device=self.device,
                )
            }

        return cams
