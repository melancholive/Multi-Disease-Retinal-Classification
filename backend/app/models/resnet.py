import torch.nn as nn
from torchvision.models import ResNet50_Weights


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_version="resnet50", pretrained=True):
        super().__init__()

        from torchvision.models import resnet50

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet50(weights=weights)
        self.feature_dim = 2048

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        features = self.resnet(x)
        return features.view(features.size(0), -1)


class ResNetMultiLabelClassifier(nn.Module):
    def __init__(self, num_classes=24, dropout=0.2, return_features=False):
        super().__init__()
        self.return_features = return_features

        # Clean backbone (raw features only)
        self.backbone = ResNetBackbone(resnet_version="resnet50", pretrained=False)
        feature_dim = self.backbone.feature_dim  # 2048 for ResNet50

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        logits = self.classifier(features)

        if self.return_features:
            return logits, features
        return logits
