import os

import torch
from dotenv import load_dotenv

from app.models.efficientnet import EfficientNetMultiLabel
from app.models.resnet import ResNetMultiLabelClassifier
from app.models.shufflenet import ShuffleNetV2MultiLabel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads paths where model checkpoints are saved
load_dotenv()
shufflenet_checkpoint_path = os.getenv("SHUFFLENET_CHECKPOINT_PATH")
if not shufflenet_checkpoint_path:
    raise OSError("Missing SHUFFLENET_CHECKPOINT_PATH in .env")

resnet_checkpoint_path = os.getenv("RESNET_CHECKPOINT_PATH")
if not resnet_checkpoint_path:
    raise OSError("Missing RESNET_CHECKPOINT_PATH in .env")

efficientnet_checkpoint_path = os.getenv("EFFICIENTNET_CHECKPOINT_PATH")
if not efficientnet_checkpoint_path:
    raise OSError("Missing EFFICIENTNET_CHECKPOINT_PATH in .env")

shufflenet_checkpoint = torch.load(
    shufflenet_checkpoint_path, map_location=device, weights_only=False
)
resnet_checkpoint = torch.load(
    resnet_checkpoint_path, map_location=device, weights_only=False
)
efficientnet_checkpoint = torch.load(
    efficientnet_checkpoint_path, map_location=device, weights_only=False
)

NUM_CLASSES = 24

shufflenet_model = ShuffleNetV2MultiLabel(num_classes=NUM_CLASSES, pretrained=False).to(
    device
)
shufflenet_model.load_state_dict(shufflenet_checkpoint)
shufflenet_model.eval()

resnet_model = ResNetMultiLabelClassifier(num_classes=NUM_CLASSES).to(device)
resnet_model.load_state_dict(resnet_checkpoint["model_state_dict"])
resnet_model.eval()

efficientnet_model = EfficientNetMultiLabel(
    num_classes=NUM_CLASSES, pretrained=False
).to(device)
efficientnet_model.load_state_dict(efficientnet_checkpoint["model_state_dict"])
efficientnet_model.eval()
