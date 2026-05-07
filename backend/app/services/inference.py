import torch
from PIL import Image

from app.models.feature_extraction import extract_features
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

    @torch.no_grad()
    def predict_from_pil(self, image: Image.Image, *, top_k: int = 5):
        image_rgb = image.convert("RGB")
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        f_shuffle = extract_features(self.shufflenet_model, image_tensor, self.device)
        f_eff = extract_features(self.efficientnet_model, image_tensor, self.device)
        f_res = extract_features(self.resnet_model, image_tensor, self.device)

        fused, weights, _, _ = self.fusion_model(f_shuffle, f_eff, f_res)
        logits = self.prediction_block(fused)
        probs = torch.sigmoid(logits).squeeze(0)

        top_k = max(1, min(int(top_k), probs.numel()))
        values, indices = torch.topk(probs, k=top_k)

        results = []
        for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
            label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            results.append(
                {"label": label, "probability": float(score), "index": int(idx)}
            )

        return {
            "top_k": results,
            "fusion_weights": weights.squeeze(0).detach().cpu().tolist(),
        }
