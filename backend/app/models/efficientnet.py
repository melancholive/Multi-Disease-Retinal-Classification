import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0


class EfficientNetMultiLabel(nn.Module):
    def __init__(
        self, num_classes=24, pretrained=True, dropout=0.2, return_features=False
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        classifier_layer = backbone.classifier[1]
        feature_dim = int(classifier_layer.in_features)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(feature_dim, num_classes)
        )
        self.return_features = return_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        if self.return_features:
            return logits, features
        return logits
