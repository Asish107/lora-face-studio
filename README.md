# LoRA Face Studio

**SDXL LoRA identity-preserving portrait generator**  
Train on 15–25 photos → Generate the same person in any outfit, place, lighting, or artistic style.

```
train 15–25 photos → LoRA (.safetensors) → generate unlimited portraits
```

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Quick Start — Mac (Local)](#quick-start--mac-local)
3. [Quick Start — Google Colab (Free GPU)](#quick-start--google-colab-free-gpu)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Generating Images](#generating-images)
7. [FastAPI App](#fastapi-app)
8. [Prompt System](#prompt-system)
9. [Style Reference](#style-reference)
10. [Reducing AI Feel](#reducing-ai-feel)
11. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

| Environment | Minimum | Recommended |
|---|---|---|
| **CUDA GPU** | 8 GB VRAM (RTX 3070) | 16–24 GB (A100, RTX 4090) |
| **Apple Silicon** | M1 16 GB unified | M2/M3 32 GB |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 30 GB free | 60 GB |
| **Training time** | ~90 min (T4) | ~30 min (A100) |
| **Inference time** | ~60 s (M2) | ~8 s (A100) |

> **Mac note:** Training is slow on CPU/MPS. Use Colab for training, Mac for inference.

---

## Quick Start — Mac (Local)

### 1. Clone & set up environment

```bash
git clone https://github.com/yourname/lora-face-studio.git
cd lora-face-studio

python3 -m venv .venv
source .venv/bin/activate

# Mac (Apple Silicon — MPS)
pip install torch torchvision
pip install -r requirements.txt
```

### 2. Prepare your images

Place **15–25 photos** of the person in `data/raw_images/`.

Photo guidelines:
- Minimum resolution: 512 × 512 (1024+ preferred)
- Variety: different lighting, angles, expressions, backgrounds
- Avoid: heavy filters, sunglasses, masked faces
- Format: JPEG or PNG

### 3. Prepare dataset

```bash
python train/dataset_prep.py \
  --source  data/raw_images \
  --output  data/dataset \
  --trigger jhndoe \
  --name    john_doe
```

This will:
- Detect and crop faces (OpenCV)
- Resize to 1024 × 1024
- Generate caption `.txt` files
- Create kohya-ss folder structure: `data/dataset/10_john_doe/`

### 4. Train LoRA (Mac — slow, use Colab instead)

```bash
# Install kohya-ss sd-scripts
git clone https://github.com/kohya-ss/sd-scripts.git
pip install -r sd-scripts/requirements.txt

# Launch training (no accelerate on Mac)
python train/train_lora.py \
  --trigger jhndoe \
  --data_dir data/dataset \
  --output output/lora \
  --no-accelerate
```

### 5. Generate images

```bash
# Single image
python inference/generate.py \
  --lora output/lora/jhndoe_lora.safetensors \
  --trigger jhndoe \
  --outfit "navy blazer, open collar shirt" \
  --location "Parisian café at dusk" \
  --lighting golden_hour \
  --style cinematic

# 6 automatic variations (grid)
python inference/generate.py \
  --lora output/lora/jhndoe_lora.safetensors \
  --trigger jhndoe \
  --batch
```

---

## Quick Start — Google Colab (Free GPU)

### One-command bootstrap

Open a new Colab notebook (select **T4 GPU** runtime) and run:

```python
!git clone https://github.com/yourname/lora-face-studio.git
%cd lora-face-studio

!python scripts/colab_train.py \
    --trigger   jhndoe \
    --name      john_doe \
    --epochs    25 \
    --dim       32
```

The script will:
1. Install all dependencies
2. Mount your Google Drive
3. Open a file picker to upload images
4. Run dataset prep
5. Run training
6. Save the `.safetensors` LoRA to your Drive

### Manual Colab steps

```python
# Cell 1 — Install
!pip install -q diffusers[torch] transformers accelerate safetensors peft
!pip install -q bitsandbytes xformers
!pip install -q Pillow opencv-python-headless pyyaml

# Cell 2 — Clone kohya
!git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git
!pip install -q -r sd-scripts/requirements.txt

# Cell 3 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 4 — Upload images
from google.colab import files
uploaded = files.upload()
import os
os.makedirs('/content/raw', exist_ok=True)
for name, data in uploaded.items():
    open(f'/content/raw/{name}', 'wb').write(data)

# Cell 5 — Dataset prep
!python train/dataset_prep.py \
    --source /content/raw \
    --output /content/dataset \
    --trigger jhndoe \
    --name john_doe \
    --no-caption

# Cell 6 — Train
!accelerate launch --num_cpu_threads_per_process 2 \
    sd-scripts/sdxl_train_network.py \
    --config_file train/config.yaml \
    --train_data_dir /content/dataset \
    --output_dir /content/drive/MyDrive/LoRA \
    --output_name jhndoe_lora

# Cell 7 — Generate
!python inference/generate.py \
    --lora /content/drive/MyDrive/LoRA/jhndoe_lora.safetensors \
    --trigger jhndoe \
    --batch
```

---

## Dataset Preparation

### Folder structure (after prep)

```
data/
  raw_images/          ← your original photos
    photo_001.jpg
    photo_002.jpg
    ...
  dataset/
    10_john_doe/       ← kohya format: <repeats>_<name>
      john_doe_0000.png
      john_doe_0000.txt   ← caption file
      john_doe_0001.png
      john_doe_0001.txt
      ...
```

### Caption file format

Each `.txt` should contain a single line:

```
photo of jhndoe, natural skin texture, subtle facial asymmetry, high detail portrait
```

The `--auto-caption` flag uses BLIP to generate richer captions automatically.

### Manual caption writing tips

- Always start with `photo of TRIGGER_WORD,`
- Describe lighting, mood, clothing if visible
- Include realism tokens: `natural skin pores`, `slight asymmetry`
- Keep under 100 tokens

### Image quality checklist

- [x] Sharp focus on the face
- [x] Multiple different backgrounds
- [x] Multiple lighting conditions
- [x] Both front-facing and three-quarter angles
- [x] At least one close-up crop and one half-body
- [x] No heavy makeup or filters on all images

---

## Training

### Config tuning guide

| Parameter | Default | Fewer images (<15) | Many images (>30) |
|---|---|---|---|
| `max_train_epochs` | 30 | 40 | 20 |
| `network_dim` | 32 | 16 | 64 |
| `learning_rate` | 1e-4 | 8e-5 | 1.2e-4 |
| `repeats` (folder) | 10 | 15 | 8 |

### Signs of under-training
- Face looks generic, doesn't match the person
- Trigger word has no effect
- → Increase epochs or repeats

### Signs of over-training
- Face is copy-pasted, artifacts appear
- Background is always the same as training images
- Style control stops working
- → Reduce epochs, reduce `lora_scale` at inference (try 0.6–0.75)

### Monitoring training

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 — watch the loss curve stabilise.

---

## Generating Images

### CLI reference

```bash
python inference/generate.py \
  --lora       path/to/lora.safetensors \   # required
  --trigger    jhndoe \                      # required
  --outfit     "black turtleneck" \
  --location   "Tokyo street at night" \
  --lighting   neon_night \
  --style      cinematic \
  --steps      35 \
  --guidance   7.5 \
  --seed       42 \
  --lora_scale 0.85 \
  --upscale \                               # 2× Lanczos upscale
  --refiner \                               # use SDXL refiner
  --batch                                   # generate 6 variations
```

### Python API

```python
from inference.generate import FaceGenerator, GenerationParams
from inference.prompt_engine import PromptConfig

params = GenerationParams(
    lora_path    = "output/lora/jhndoe_lora.safetensors",
    trigger_word = "jhndoe",
    lora_scale   = 0.85,
    steps        = 35,
)

gen = FaceGenerator(params)

cfg = PromptConfig(
    trigger_word = "jhndoe",
    outfit       = "red leather jacket",
    location     = "neon-lit Tokyo alley",
    lighting     = "neon_night",
    style        = "cinematic",
    seed         = 42,
)

path = gen.generate_single(cfg)
print(f"Saved to {path}")
```

---

## FastAPI App

### Start server

```bash
# Set environment variables first
export LORA_PATH="output/lora/jhndoe_lora.safetensors"
export TRIGGER_WORD="jhndoe"
export LORA_SCALE="0.85"
# Optional:
# export LOW_VRAM=1          # enable on Mac / low VRAM GPU
# export USE_REFINER=1       # use SDXL refiner for polish

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the web UI.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness check |
| `GET` | `/styles` | List styles |
| `GET` | `/lightings` | List lighting presets |
| `POST` | `/generate` | Generate single image |
| `POST` | `/batch` | Generate multiple (returns grid) |
| `POST` | `/load` | Swap LoRA at runtime |

### Example API call

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "outfit":  "grey wool coat",
    "location": "snowy mountain path",
    "lighting": "foggy_morning",
    "style":   "cinematic",
    "steps":   35,
    "seed":    42
  }' | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
open('result.png','wb').write(base64.b64decode(d['image_b64']))
print('Saved result.png')
"
```

---

## Prompt System

### Identity anchor (always present)

```
portrait of jhndoe, <lora:jhndoe:0.85>, same person, consistent face, consistent identity
```

### Full positive prompt structure

```
[IDENTITY ANCHOR], [OUTFIT], [LOCATION], [CAMERA ANGLE], [LIGHTING], [STYLE TOKENS], [REALISM TOKENS], highly detailed, sharp focus, masterwork, human-made
```

### Full negative prompt

```
face distortion, identity drift, different person, face swap,
extra fingers, malformed hands, fused fingers, extra limbs,
duplicate, cloned face,
cgi, 3d render, 3d model, ai artifacts, digital painting look,
over-smooth skin, plastic skin, rubber skin, airbrushed,
overly symmetrical face, uncanny valley,
blurry, overexposed, underexposed, jpeg artifacts, watermark,
text, logo, cropped head, bad framing
```

### LoRA scale guide

| Scale | Effect |
|---|---|
| `0.6–0.7` | Loose similarity, maximum style freedom |
| `0.75–0.85` | **Recommended** — strong identity + style flexibility |
| `0.9–1.0` | Strict identity, may resist strong style changes |
| `>1.0` | Over-fitting artifacts |

### Seed control

```python
# Reproducible
cfg.seed = 42

# Explore variations on the same prompt
for seed in [42, 123, 999, 2024]:
    cfg.seed = seed
    gen.generate_single(cfg)
```

---

## Style Reference

| Style key | Description | Best for |
|---|---|---|
| `hyperrealistic` | Sony A7R V, natural skin detail | Portraits, headshots |
| `cinematic` | Film grain, anamorphic, Kodak palette | Dramatic storytelling |
| `fashion_photography` | Vogue editorial, seamless | Outfit showcases |
| `street_photography` | Leica M, candid, documentary | Urban, authentic |
| `charcoal` | Rough paper, tonal study | Artistic portraits |
| `watercolor` | Wet-on-wet, visible bleeds | Soft artistic feel |
| `anime` | Ghibli-inspired, cel-shading | Animated aesthetic |
| `oil_painting` | Old masters, impasto texture | Classical portrait |

---

## Reducing AI Feel

This pipeline uses multiple layers to push output away from the generic "AI look":

### 1. Prompt-level micro-imperfections

The prompt engine randomly injects vocabulary like:
```
natural skin pores, subtle asymmetry of features, natural under-eye shadows,
real hair flyaways, uneven lip line, micro skin imperfections,
natural skin tone variation, authentic eye moisture, slight chromatic aberration,
natural lens vignetting, subtle film grain
```

### 2. Post-processing pipeline (`utils/image_utils.py`)

Applied automatically to every output:

```python
img = subtle_sharpening(img)    # unsharp mask — reveals micro-detail
img = add_film_grain(img)       # luminance noise at strength=0.018
img = add_vignette(img)         # optical lens fall-off
img = colour_grade(img)         # mild contrast + slight desaturation
```

### 3. Training config choices

- `noise_offset: 0.0357` — improves exposure range, reduces flat tones
- `multires_noise_iterations: 6` — adds frequency variety to noise
- `min_snr_gamma: 5.0` — stable loss weighting, avoids over-smoothing
- `flip_aug: true` — prevents mirror-image memorisation

### 4. Prompt writing tips

**Do:**
- Specify exact lens: `85mm f/1.4`, `35mm f/2`, `50mm f/1.8`
- Name real photographers' styles: `in the style of Annie Leibovitz`
- Use time-of-day: `golden hour`, `blue hour`, `noon harsh light`
- Add imperfection context: `slightly tired eyes`, `natural sun-kissed skin`

**Avoid:**
- `best quality, masterpiece, 8k` — these bias toward AI-smooth outputs
- `perfect skin, flawless` — removes natural texture
- Very short prompts — more context = more realistic

---

## Troubleshooting

### Face doesn't look like the person
- Increase `lora_scale` to 0.9–1.0
- Increase training epochs by 10
- Check captions all start with the trigger word
- Verify images were properly cropped (run with `--no-crop` to skip if already portrait-cropped)

### "No face detected" during dataset prep
- Use `--no-crop` if images are already well-framed portraits
- Install full OpenCV: `pip install opencv-python` (not headless)

### Out of memory (CUDA OOM)
- Set `LOW_VRAM=1` in environment
- Reduce `train_batch_size` to 1 in `config.yaml`
- Add `--low_vram` to training command
- Enable `gradient_checkpointing: true` (already default)

### Slow inference on Mac
- Mac MPS doesn't support fp16 — inference uses fp32 (2× slower but correct)
- Expected: ~40–90s per image on M1/M2
- For faster iteration, set `--steps 20`

### LoRA not loading
- Ensure the `.safetensors` file is not corrupted: `python -c "from safetensors import safe_open; safe_open('your.safetensors', framework='pt')"`
- Match trigger word exactly (case-sensitive)
- Try `lora_scale=0.75` if outputs look distorted

### kohya-ss not found
- The training script auto-clones it; ensure git is installed: `which git`
- Manually clone: `git clone https://github.com/kohya-ss/sd-scripts.git`

### TensorBoard shows NaN loss
- Reduce learning rate: `unet_lr: 5e-5`
- Increase `lr_warmup_steps` to 100
- Check for corrupted images in dataset

---

## Project Structure

```
lora-face-studio/
├── train/
│   ├── dataset_prep.py     # image crop, resize, caption, kohya structure
│   ├── train_lora.py       # training launcher (validates, patches config, launches)
│   ├── config.yaml         # full kohya-ss SDXL LoRA training config
│   └── sample_prompts.txt  # prompts used to preview training progress
├── inference/
│   ├── generate.py         # main generation API + CLI
│   ├── prompt_engine.py    # style/lighting/outfit prompt builder
│   └── model_loader.py     # SDXL + LoRA pipeline manager
├── utils/
│   └── image_utils.py      # resize, grain, vignette, upscale, grid
├── app/
│   └── main.py             # FastAPI REST API + minimal web UI
├── scripts/
│   └── colab_train.py      # Google Colab one-command bootstrap
├── data/
│   ├── raw_images/         # put your photos here
│   ├── dataset/            # prepared by dataset_prep.py
│   └── outputs/            # generated images saved here
├── output/
│   └── lora/               # trained .safetensors files saved here
├── requirements.txt
└── README.md
```

---

## License

This project is released under the MIT License.  
The SDXL base model is subject to the [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).
