"""
app/main.py
-----------
FastAPI application exposing the LoRA face generator as a REST API.

Endpoints:
  POST /generate   — generate a single image
  POST /batch      — generate multiple variations
  GET  /styles     — list available styles
  GET  /lightings  — list available lighting presets
  GET  /health     — liveness check

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Environment variables (set before starting):
    LORA_PATH        path to .safetensors file (required)
    TRIGGER_WORD     LoRA trigger word          (required)
    LORA_SCALE       float, default 0.85
    BASE_MODEL       HF model id or local path
    USE_REFINER      0 or 1
    LOW_VRAM         0 or 1  (enables CPU offload)
"""

import os
import io
import base64
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from PIL import Image

from inference.model_loader import SDXLModelLoader
from inference.generate import FaceGenerator, GenerationParams
from inference.prompt_engine import PromptConfig, STYLES, LIGHTING_PRESETS
from utils.image_utils import apply_realism_postprocess, to_bytes, make_grid


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "LoRA Face Generator",
    description = "SDXL + LoRA identity-preserving portrait generator",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────
# Startup / global state
# ─────────────────────────────────────────────

_generator: Optional[FaceGenerator] = None


@app.on_event("startup")
async def startup() -> None:
    global _generator

    lora_path    = os.environ.get("LORA_PATH")
    trigger_word = os.environ.get("TRIGGER_WORD")

    if not lora_path or not trigger_word:
        logger.warning(
            "LORA_PATH or TRIGGER_WORD not set — /generate will return 503 "
            "until a LoRA is loaded via /load"
        )
        return

    _init_generator(lora_path, trigger_word)


def _init_generator(lora_path: str, trigger_word: str) -> None:
    global _generator
    params = GenerationParams(
        lora_path       = lora_path,
        trigger_word    = trigger_word,
        lora_scale      = float(os.environ.get("LORA_SCALE", 0.85)),
        use_refiner     = bool(int(os.environ.get("USE_REFINER", 0))),
        low_vram        = bool(int(os.environ.get("LOW_VRAM", 0))),
    )
    _generator = FaceGenerator(params)
    logger.info("Generator ready — trigger: '%s'", trigger_word)


# ─────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    outfit:         str   = Field("casual everyday clothing", example="navy blazer, open collar shirt")
    location:       str   = Field("natural outdoor setting",  example="Tokyo street at night")
    lighting:       str   = Field("overcast",                 example="golden_hour")
    style:          str   = Field("hyperrealistic",           example="cinematic")
    camera_angle:   str   = Field("eye-level portrait",       example="slight upward angle, three-quarter view")
    extra_positive: str   = Field("",                         example="")
    extra_negative: str   = Field("",                         example="")
    seed:           Optional[int] = None
    steps:          int   = Field(35, ge=10, le=80)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    lora_scale:     float = Field(0.85, ge=0.0, le=1.5)
    realism_post:   bool  = True    # apply grain/vignette post-processing
    output_format:  str   = "base64"  # "base64" | "url" (url not yet impl)

    @validator("lighting")
    def lighting_valid(cls, v):
        if v not in LIGHTING_PRESETS:
            raise ValueError(f"lighting must be one of: {list(LIGHTING_PRESETS.keys())}")
        return v

    @validator("style")
    def style_valid(cls, v):
        if v not in STYLES:
            raise ValueError(f"style must be one of: {list(STYLES.keys())}")
        return v


class LoadLoraRequest(BaseModel):
    lora_path:    str
    trigger_word: str
    lora_scale:   float = 0.85


class GenerateResponse(BaseModel):
    image_b64:   str
    positive:    str
    negative:    str
    seed:        Optional[int]
    style:       str
    timing_ms:   int


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _require_generator():
    if _generator is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Generator not initialised. Set LORA_PATH + TRIGGER_WORD env vars or POST /load."
        )


def _img_to_b64(img: Image.Image) -> str:
    return base64.b64encode(to_bytes(img, "PNG")).decode("utf-8")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "generator_ready": _generator is not None}


@app.get("/styles")
async def list_styles():
    return {"styles": list(STYLES.keys())}


@app.get("/lightings")
async def list_lightings():
    return {"lightings": list(LIGHTING_PRESETS.keys())}


@app.post("/load")
async def load_lora(req: LoadLoraRequest):
    """Dynamically load (or swap) a LoRA at runtime."""
    try:
        _init_generator(req.lora_path, req.trigger_word)
        return {"status": "ok", "trigger": req.trigger_word}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    import time
    _require_generator()
    t0 = time.time()

    # Override per-request lora_scale
    _generator.params.guidance_scale = req.guidance_scale
    _generator.params.steps          = req.steps

    cfg = PromptConfig(
        trigger_word    = _generator.params.trigger_word,
        lora_weight     = req.lora_scale,
        outfit          = req.outfit,
        location        = req.location,
        lighting        = req.lighting,
        style           = req.style,
        camera_angle    = req.camera_angle,
        extra_positive  = req.extra_positive,
        extra_negative  = req.extra_negative,
        seed            = req.seed,
        realism_tokens  = 4,
    )

    positive, negative = _generator.engine.build(cfg)

    try:
        out_path = _generator.generate_single(cfg)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(500, f"Generation failed: {e}")

    img = Image.open(out_path)
    if req.realism_post:
        img = apply_realism_postprocess(img)

    elapsed = int((time.time() - t0) * 1000)

    return GenerateResponse(
        image_b64  = _img_to_b64(img),
        positive   = positive,
        negative   = negative,
        seed       = req.seed,
        style      = req.style,
        timing_ms  = elapsed,
    )


@app.post("/batch")
async def batch_generate(requests: List[GenerateRequest]):
    """Generate multiple variations, returns a grid PNG in base64."""
    _require_generator()

    if len(requests) > 8:
        raise HTTPException(400, "Maximum 8 images per batch request")

    images = []
    for req in requests:
        cfg = PromptConfig(
            trigger_word   = _generator.params.trigger_word,
            lora_weight    = req.lora_scale,
            outfit         = req.outfit,
            location       = req.location,
            lighting       = req.lighting,
            style          = req.style,
            camera_angle   = req.camera_angle,
            seed           = req.seed,
        )
        out_path = _generator.generate_single(cfg)
        img = Image.open(out_path)
        if req.realism_post:
            img = apply_realism_postprocess(img)
        images.append(img)

    grid = make_grid(images)
    return {"grid_b64": _img_to_b64(grid), "count": len(images)}


# ─────────────────────────────────────────────
# Minimal web UI
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    styles    = list(STYLES.keys())
    lightings = list(LIGHTING_PRESETS.keys())

    style_opts    = "\n".join(f'<option value="{s}">{s}</option>' for s in styles)
    lighting_opts = "\n".join(f'<option value="{l}">{l}</option>' for l in lightings)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LoRA Face Generator</title>
  <style>
    :root {{ --bg:#0d0d0f; --panel:#16161a; --accent:#c8a96e; --text:#e8e8e8; --border:#2a2a2e; }}
    *{{ box-sizing:border-box; margin:0; padding:0 }}
    body{{ background:var(--bg); color:var(--text); font-family:'Georgia',serif; min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:40px 20px }}
    h1{{ font-size:2rem; letter-spacing:.12em; color:var(--accent); margin-bottom:8px }}
    .subtitle{{ color:#888; font-size:.85rem; margin-bottom:36px; letter-spacing:.06em }}
    .container{{ display:grid; grid-template-columns:380px 1fr; gap:32px; width:100%; max-width:1100px }}
    .panel{{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:28px }}
    label{{ display:block; font-size:.78rem; letter-spacing:.08em; color:#aaa; margin-bottom:6px; margin-top:18px; text-transform:uppercase }}
    input,select,textarea{{ width:100%; background:#1e1e22; border:1px solid var(--border); color:var(--text); border-radius:6px; padding:10px 12px; font-size:.9rem }}
    textarea{{ height:80px; resize:vertical }}
    button{{ margin-top:24px; width:100%; padding:14px; background:var(--accent); color:#0d0d0f; border:none; border-radius:8px; font-size:1rem; letter-spacing:.1em; cursor:pointer; font-weight:700; transition:opacity .2s }}
    button:hover{{ opacity:.85 }}
    button:disabled{{ opacity:.4; cursor:not-allowed }}
    #output{{ display:flex; flex-direction:column; align-items:center; justify-content:center }}
    #output img{{ max-width:100%; border-radius:10px; box-shadow:0 8px 40px rgba(0,0,0,.6) }}
    #status{{ font-size:.82rem; color:#888; margin-top:12px; min-height:20px }}
    #prompts{{ margin-top:20px; width:100% }}
    #prompts details{{ margin-top:10px }}
    #prompts summary{{ cursor:pointer; color:#aaa; font-size:.78rem; letter-spacing:.06em }}
    pre{{ background:#111; padding:12px; border-radius:6px; font-size:.72rem; white-space:pre-wrap; color:#ccc; margin-top:8px }}
    .spinner{{ width:40px; height:40px; border:3px solid #333; border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; margin:60px auto }}
    @keyframes spin{{ to{{transform:rotate(360deg)}} }}
  </style>
</head>
<body>
  <h1>LoRA Face Generator</h1>
  <p class="subtitle">SDXL · Identity Preservation · Realistic Output</p>
  <div class="container">
    <div class="panel">
      <label>Outfit</label>
      <input id="outfit" value="navy blazer, open collar white shirt" />
      <label>Location</label>
      <input id="location" value="Parisian café terrace at dusk" />
      <label>Lighting</label>
      <select id="lighting">{lighting_opts}</select>
      <label>Style</label>
      <select id="style">{style_opts}</select>
      <label>Camera angle</label>
      <input id="angle" value="eye-level portrait, three-quarter view" />
      <label>Seed (blank = random)</label>
      <input id="seed" type="number" placeholder="e.g. 42" />
      <label>Steps</label>
      <input id="steps" type="number" value="35" min="10" max="80" />
      <label>Guidance Scale (CFG)</label>
      <input id="cfg" type="number" step="0.5" value="7.5" min="1" max="20" />
      <button id="btn" onclick="generate()">Generate Portrait</button>
    </div>
    <div id="output" class="panel">
      <p style="color:#555;font-size:.9rem">Your portrait will appear here</p>
    </div>
  </div>
  <script>
    async function generate() {{
      const btn = document.getElementById('btn');
      const out = document.getElementById('output');
      btn.disabled = true;
      out.innerHTML = '<div class="spinner"></div><p id="status">Generating — this may take 30–90 seconds…</p>';
      const payload = {{
        outfit:        document.getElementById('outfit').value,
        location:      document.getElementById('location').value,
        lighting:      document.getElementById('lighting').value,
        style:         document.getElementById('style').value,
        camera_angle:  document.getElementById('angle').value,
        seed:          document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
        steps:         parseInt(document.getElementById('steps').value),
        guidance_scale:parseFloat(document.getElementById('cfg').value),
      }};
      try {{
        const res  = await fetch('/generate', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload) }});
        const data = await res.json();
        if (!res.ok) {{ out.innerHTML = `<p style="color:#e55">Error: ${{data.detail}}</p>`; return; }}
        out.innerHTML = `
          <img src="data:image/png;base64,${{data.image_b64}}" />
          <p id="status">✓ Generated in ${{(data.timing_ms/1000).toFixed(1)}}s · Style: ${{data.style}}</p>
          <div id="prompts">
            <details><summary>▸ Positive prompt</summary><pre>${{data.positive}}</pre></details>
            <details><summary>▸ Negative prompt</summary><pre>${{data.negative}}</pre></details>
          </div>`;
      }} catch(e) {{
        out.innerHTML = `<p style="color:#e55">Request failed: ${{e.message}}</p>`;
      }} finally {{
        btn.disabled = false;
      }}
    }}
  </script>
</body>
</html>"""
