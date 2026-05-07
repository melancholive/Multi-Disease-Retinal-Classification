import os

import torch
from dotenv import load_dotenv

from app.models.transformer import PredictionBlock, TransformerAttentionFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
transformer_checkpoint_path = os.getenv("TRANSFORMER_CHECKPOINT_PATH")
if not transformer_checkpoint_path:
    raise OSError("Missing TRANSFORMER_CHECKPOINT_PATH in .env")

fusion_model = TransformerAttentionFusion().to(device)
prediction_block = PredictionBlock().to(device)

checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
fusion_model.load_state_dict(checkpoint["fusion_model_state_dict"])
prediction_block.load_state_dict(checkpoint["prediction_block_state_dict"])

fusion_model.eval()
prediction_block.eval()
