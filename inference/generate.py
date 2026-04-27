"""
generate.py
-----------
High-level generation API:
  • Single image  — generate_single()
  • Batch (grid)  — generate_batch()
  • CLI           — python inference/generate.py --lora ./output/lora/x.safetensors

All outputs are saved to ./data/outputs/ with metadata JSON sidecar.
"""

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from inference.model_loader import SDXLModelLoader, get_device
from inference.prompt_engine import PromptEngine, PromptConfig, STYLES, LIGHTING_PRESETS

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Generation parameters dataclass
# ─────────────────────────────────────────────

from dataclasses import dataclass, field as dc_field

@dataclass
class GenerationParams:
    lora_path:      str
    trigger_word:   str
    lora_scale:     float = 0.85      # sweet spot: strong identity, not over-fitted
    width:          int   = 1024
    height:         int   = 1024
    steps:          int   = 35
    guidance_scale: float = 7.5       # CFG — higher = more prompt adherent
    seed:           Optional[int] = None
    # Refiner
    use_refiner:    bool  = False
    refiner_model:  str   = "stabilityai/stable-diffusion-xl-refiner-1.0"
    refiner_strength: float = 0.30    # how much refiner polishes the base output
    # Upscaling (PIL-based, free)
    upscale:        bool  = False
    upscale_factor: int   = 2


# ─────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────

class FaceGenerator:

    def __init__(
        self,
        params:  GenerationParams,
        loader:  Optional[SDXLModelLoader] = None,
    ):
        self.params = params
        self.device = get_device()

        # Reuse an already-loaded pipeline, or create a fresh one
        if loader is None:
            self.loader = SDXLModelLoader(
                refiner_model = params.refiner_model if params.use_refiner else None,
                low_vram      = (self.device.type != "cuda"),
            ).load()
        else:
            self.loader = loader

        self.loader.load_lora(
            lora_path    = params.lora_path,
            lora_scale   = params.lora_scale,
        )

        self.engine = PromptEngine(
            trigger_word = params.trigger_word,
            lora_weight  = params.lora_scale,
        )

    # ── Core inference ────────────────────────────────────────────────────

    def _make_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(seed)
        return gen

    def _run_base(
        self,
        positive: str,
        negative: str,
        seed:     Optional[int],
    ) -> Image.Image:
        p = self.params
        gen = self._make_generator(seed)

        kwargs = dict(
            prompt              = positive,
            negative_prompt     = negative,
            width               = p.width,
            height              = p.height,
            num_inference_steps = p.steps,
            guidance_scale      = p.guidance_scale,
            generator           = gen,
        )

        if p.use_refiner:
            kwargs["output_type"]       = "latent"
            kwargs["denoising_end"]     = 1.0 - p.refiner_strength

        output = self.loader.pipe(**kwargs)
        return output.images[0]

    def _run_refiner(
        self,
        latent:   "torch.Tensor",
        positive: str,
        negative: str,
        seed:     Optional[int],
    ) -> Image.Image:
        gen = self._make_generator(seed)
        output = self.loader.refiner_pipe(
            prompt              = positive,
            negative_prompt     = negative,
            image               = latent,
            num_inference_steps = self.params.steps,
            denoising_start     = 1.0 - self.params.refiner_strength,
            guidance_scale      = self.params.guidance_scale,
            generator           = gen,
        )
        return output.images[0]

    def _maybe_upscale(self, img: Image.Image) -> Image.Image:
        if not self.params.upscale:
            return img
        factor = self.params.upscale_factor
        new_w  = img.width  * factor
        new_h  = img.height * factor
        return img.resize((new_w, new_h), Image.LANCZOS)

    # ── Metadata sidecar ─────────────────────────────────────────────────

    @staticmethod
    def _save_metadata(path: Path, positive: str, negative: str, params: GenerationParams, cfg: PromptConfig) -> None:
        meta = {
            "timestamp":     datetime.now().isoformat(),
            "lora":          params.lora_path,
            "trigger":       params.trigger_word,
            "lora_scale":    params.lora_scale,
            "steps":         params.steps,
            "guidance_scale":params.guidance_scale,
            "seed":          cfg.seed,
            "style":         cfg.style,
            "outfit":        cfg.outfit,
            "location":      cfg.location,
            "lighting":      cfg.lighting,
            "positive":      positive,
            "negative":      negative,
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    # ── Public: generate single ───────────────────────────────────────────

    def generate_single(
        self,
        cfg:      PromptConfig,
        out_dir:  Path = OUTPUT_DIR,
    ) -> Path:
        positive, negative = self.engine.build(cfg)

        logger.info("Generating — style: %s | outfit: %s | location: %s",
                    cfg.style, cfg.outfit[:40], cfg.location[:40])
        logger.debug("Positive: %s", positive[:120])

        # Resolve seed
        seed = cfg.seed if cfg.seed is not None else None
        result = self._run_base(positive, negative, seed)

        if self.params.use_refiner and self.loader.refiner_pipe:
            result = self._run_refiner(result, positive, negative, seed)

        result = self._maybe_upscale(result)

        # Save
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{cfg.trigger_word}_{cfg.style}_{ts}"
        out_path = out_dir / f"{stem}.png"
        result.save(out_path, format="PNG")
        self._save_metadata(out_path, positive, negative, self.params, cfg)

        logger.info("Saved → %s", out_path)
        return out_path

    # ── Public: batch grid ────────────────────────────────────────────────

    def generate_batch(
        self,
        configs:  list[PromptConfig],
        out_dir:  Path = OUTPUT_DIR,
        save_grid: bool = True,
    ) -> list[Path]:
        paths  = []
        images = []

        for cfg in configs:
            p = self.generate_single(cfg, out_dir)
            paths.append(p)
            images.append(Image.open(p))

        if save_grid and images:
            grid_path = self._save_grid(images, out_dir)
            logger.info("Grid saved → %s", grid_path)

        return paths

    @staticmethod
    def _save_grid(images: list[Image.Image], out_dir: Path) -> Path:
        import math
        n    = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))

        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            grid.paste(img, (c * w, r * h))

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"grid_{ts}.png"
        grid.save(path, format="PNG")
        return path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

EXAMPLE_VARIATIONS = [
    # (outfit,                                   location,                       lighting,      style)
    ("tailored navy blazer, open collar",        "Parisian café terrace",        "golden_hour", "cinematic"),
    ("white linen shirt",                        "mediterranean rooftop at dusk","neon_night",  "hyperrealistic"),
    ("casual denim jacket, grey tee",            "urban alleyway, Tokyo",        "overcast",    "street_photography"),
    ("black turtleneck",                         "minimalist studio",            "studio_soft", "fashion_photography"),
    ("heavy wool coat, scarf",                   "snowy mountain path",          "foggy_morning","hyperrealistic"),
    ("summer floral shirt",                      "tropical beach, sunset",       "golden_hour", "cinematic"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate face images with SDXL LoRA")
    parser.add_argument("--lora",       required=True,  help="Path to .safetensors LoRA file")
    parser.add_argument("--trigger",    required=True,  help="LoRA trigger word")
    parser.add_argument("--outfit",     default=None,   help="Outfit description")
    parser.add_argument("--location",   default=None,   help="Location/setting")
    parser.add_argument("--lighting",   default="overcast",
                        choices=list(LIGHTING_PRESETS.keys()))
    parser.add_argument("--style",      default="hyperrealistic",
                        choices=list(STYLES.keys()))
    parser.add_argument("--steps",      type=int,   default=35)
    parser.add_argument("--guidance",   type=float, default=7.5)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--lora_scale", type=float, default=0.85)
    parser.add_argument("--upscale",    action="store_true")
    parser.add_argument("--refiner",    action="store_true")
    parser.add_argument("--batch",      action="store_true",
                        help="Generate 6 example variations")
    parser.add_argument("--output",     default="./data/outputs")
    args = parser.parse_args()

    params = GenerationParams(
        lora_path       = args.lora,
        trigger_word    = args.trigger,
        lora_scale      = args.lora_scale,
        steps           = args.steps,
        guidance_scale  = args.guidance,
        upscale         = args.upscale,
        use_refiner     = args.refiner,
    )

    gen = FaceGenerator(params)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        configs = [
            PromptConfig(
                trigger_word = args.trigger,
                outfit       = outfit,
                location     = location,
                lighting     = lighting,
                style        = style,
                seed         = args.seed,
            )
            for outfit, location, lighting, style in EXAMPLE_VARIATIONS
        ]
        gen.generate_batch(configs, out_dir)
    else:
        cfg = PromptConfig(
            trigger_word = args.trigger,
            outfit       = args.outfit or "casual everyday clothing",
            location     = args.location or "natural outdoor setting",
            lighting     = args.lighting,
            style        = args.style,
            seed         = args.seed,
        )
        gen.generate_single(cfg, out_dir)


if __name__ == "__main__":
    main()
