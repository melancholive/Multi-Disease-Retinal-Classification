import torch.nn as nn
from torchvision import models
from torchvision.models import ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2MultiLabel(nn.Module):
    def __init__(
        self, num_classes=24, pretrained=True, dropout=0.2, return_features=False
    ):
        super().__init__()
        self.return_features = return_features

        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        backbone = models.shufflenet_v2_x1_0(weights=weights)

        in_features = backbone.fc.in_features
        # torch/torchvision type stubs consider this a Linear; runtime allows Identity.
        backbone.fc = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # [B, 1024]
        logits = self.classifier(self.dropout(features))

        if self.return_features:
            return logits, features
        return logits
