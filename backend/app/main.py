from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.services.inference import InferenceService
from app.services.load_cnn import (
    device as cnn_device,
)
from app.services.load_cnn import (
    efficientnet_model,
    resnet_model,
    shufflenet_model,
)
from app.services.load_transformer import fusion_model, prediction_block

CLASS_NAMES = [
    "CENTRAL_SEROUS_RETINOPATHY",
    "DIABETIC_RETINOPATHY",
    "GLAUCOMA",
    "NORMAL",
    "MACULAR_SCAR",
    "MYOPIA",
    "PTERYGIUM",
    "RETINAL_DETACHMENT",
    "RETINITIS_PIGMENTOSA",
    "HYPERTENSION",
    "CATARACT",
    "AGE_RELATED_MACULAR_DEGENERATION",
    "DISC_EDEMA",
    "MEDIA_HAZE",
    "OPTIC_DISC_CUPPING",
    "TESSELLATION",
    "DRUSEN",
    "BRANCH_RETINAL_VEIN_OCCLUSION",
    "OPTIC_DISC_PALLOR",
    "CENTRAL_RETINAL_VEIN_OCCLUSION",
    "CHOROIDAL_NEOVASCULARIZATION",
    "RETINITIS",
    "LASER_SCARS",
    "ARTERIOSCLEROTIC_RETINOPATHY",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferenceService(
        shufflenet_model=shufflenet_model,
        efficientnet_model=efficientnet_model,
        resnet_model=resnet_model,
        fusion_model=fusion_model,
        prediction_block=prediction_block,
        device=cnn_device,
        class_names=CLASS_NAMES,
    )
    yield

    app.state.inference = None


app = FastAPI(title="Fusion Transformer Inference", lifespan=lifespan)

# Local dev: allow the Next.js dev server to call the API directly.
# If you only ever call through Next.js route handlers (server-to-server), CORS is not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
