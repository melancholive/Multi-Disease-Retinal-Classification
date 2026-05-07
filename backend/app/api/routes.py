from io import BytesIO

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

router = APIRouter()


@router.get("/health")
def health(request: Request):
    device = getattr(request.app.state.inference, "device", None)
    return {
        "status": "ok",
        "device": str(device) if device is not None else None,
    }


@router.post("/predict")
async def predict(request: Request, image: UploadFile = File(...), top_k: int = 5):
    contents = await image.read()
    pil = Image.open(BytesIO(contents))
    return request.app.state.inference.predict_from_pil(pil, top_k=top_k)
