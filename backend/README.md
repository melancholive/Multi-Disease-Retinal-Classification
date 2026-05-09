# Backend

FastAPI backend for retinal disease prediction, Grad-CAM generation, and the agent
API used by the frontend.

## Requirements

- Python 3.11
- Model checkpoint files for the CNNs and fusion transformer
- An OpenAI API key for the agent LLM
- Optional: a remote MedGemma API URL if you want the tools to use MedGemma without
  loading it locally

## Setup

From the repository root:

```bash
cd backend
```

Install dependencies with `uv`:

```bash
uv sync
```

## Environment

Create `backend/.env`:

```bash
SHUFFLENET_CHECKPOINT_PATH=checkpoints/path-to-shufflenet.pt
RESNET_CHECKPOINT_PATH=checkpoints/path-to-resnet.pt
EFFICIENTNET_CHECKPOINT_PATH=checkpoints/path-to-efficientnet.pt
TRANSFORMER_CHECKPOINT_PATH=checkpoints/path-to-transformer.pt

OPENAI_API_KEY=your-openai-api-key

# Optional. Recommended if running MedGemma on Colab or another GPU machine.
MEDGEMMA_API_URL=https://your-ngrok-url.ngrok-free.app/generate
MEDGEMMA_API_TIMEOUT=600

# Optional. Only use local MedGemma warmup if your machine has enough RAM/VRAM.
WARMUP_MEDGEMMA=false
```

The checkpoint paths may be relative to `backend/` or absolute paths.

If `MEDGEMMA_API_URL` is set, the backend calls that API for MedGemma tool
responses. If it is not set, the backend falls back to loading
`google/medgemma-4b-it` locally on first use, which can require a lot of memory.

## Run

With `uv`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

With an activated virtualenv:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```


## Remote MedGemma

For Colab/ngrok, set:

```bash
MEDGEMMA_API_URL=https://your-ngrok-url.ngrok-free.app/generate
```


## Notes

- The CNN and transformer checkpoints are loaded when the backend starts.
- The agent graph is created lazily on the first agent request.
- Agent memory is in-process only. Restarting the backend clears previous agent
  sessions.
- Local MedGemma on a 16 GB machine without CUDA is usually too slow and memory
  heavy for interactive use. Prefer a remote GPU-backed MedGemma API.
