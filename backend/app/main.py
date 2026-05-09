import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
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

load_dotenv()

logger = logging.getLogger(__name__)

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


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


async def warmup_medgemma() -> None:
    from app.services.medgemma import get_medgemma_pipe

    try:
        logger.info("Starting MedGemma warmup")
        await asyncio.to_thread(get_medgemma_pipe)
        logger.info("MedGemma warmup complete")
    except Exception:
        logger.exception("MedGemma warmup failed")


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

    app.state.medgemma_warmup_task = None
    if _env_flag_enabled("WARMUP_MEDGEMMA"):
        app.state.medgemma_warmup_task = asyncio.create_task(warmup_medgemma())

    yield

    medgemma_warmup_task = app.state.medgemma_warmup_task
    if medgemma_warmup_task and not medgemma_warmup_task.done():
        medgemma_warmup_task.cancel()

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
