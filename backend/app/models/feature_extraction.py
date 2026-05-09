import torch


@torch.no_grad()
def extract_features(model, image, device):
    model.eval()
    prev_return_features = getattr(model, "return_features", None)
    if prev_return_features is not None:
        model.return_features = True

    images = image.to(device)
    if images.dim() == 3:
        images = images.unsqueeze(0)  # [1, 3, H, W]

    try:
        _, features = model(images)
        return features
    finally:
        if prev_return_features is not None:
            model.return_features = prev_return_features
