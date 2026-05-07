import torch


@torch.no_grad()
def extract_features(model, image, device):
    model.eval()
    model.return_features = True

    images = image.to(device)
    if images.dim() == 3:
        images = images.unsqueeze(0)  # [1, 3, H, W]

    _, features = model(images)
    return features
