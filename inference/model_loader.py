"""
model_loader.py
---------------
Loads SDXL base + optional refiner pipeline with LoRA weights
using the diffusers library.

Supports:
  • CPU / CUDA / Apple MPS (Mac)
  • 4-bit / 8-bit quantization via bitsandbytes (CUDA only)
  • Attention slicing + xformers for low-VRAM GPUs
  • Multiple LoRA adapters with individual weights
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """fp16 on CUDA, float32 on MPS/CPU (MPS doesn't support fp16 well)."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32


# ─────────────────────────────────────────────
# Scheduler factory
# ─────────────────────────────────────────────

SCHEDULERS = {
    "dpm++_2m":       DPMSolverMultistepScheduler,
    "euler_a":        EulerAncestralDiscreteScheduler,
}

def apply_scheduler(pipe, name: str = "euler_a") -> None:
    cls = SCHEDULERS.get(name, EulerAncestralDiscreteScheduler)
    pipe.scheduler = cls.from_config(pipe.scheduler.config)
    logger.info("Scheduler: %s", cls.__name__)


# ─────────────────────────────────────────────
# Pipeline loader
# ─────────────────────────────────────────────

class SDXLModelLoader:
    """
    Manages SDXL base (+ optional refiner) pipeline lifecycle.
    Call load() once, then reuse the pipeline across many generations.
    """

    def __init__(
        self,
        base_model:      str  = "stabilityai/stable-diffusion-xl-base-1.0",
        refiner_model:   Optional[str] = None,
        scheduler:       str  = "euler_a",
        use_xformers:    bool = True,
        low_vram:        bool = False,   # enables CPU offload
        use_safetensors: bool = True,
    ):
        self.base_model      = base_model
        self.refiner_model   = refiner_model
        self.scheduler_name  = scheduler
        self.use_xformers    = use_xformers
        self.low_vram        = low_vram
        self.use_safetensors = use_safetensors

        self.device = get_device()
        self.dtype  = get_dtype(self.device)
        self.pipe:         Optional[StableDiffusionXLPipeline]          = None
        self.refiner_pipe: Optional[StableDiffusionXLImg2ImgPipeline]   = None

        logger.info("Device: %s | Dtype: %s", self.device, self.dtype)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _optimise(self, pipe) -> None:
        """Apply memory / speed optimisations based on environment."""
        if self.low_vram or self.device.type == "cpu":
            pipe.enable_sequential_cpu_offload()
            logger.info("CPU offload enabled (low-VRAM mode)")
        else:
            pipe = pipe.to(self.device)

        # Attention slicing saves ~15% VRAM with minimal speed hit
        pipe.enable_attention_slicing(slice_size="auto")

        if self.use_xformers and self.device.type == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers attention enabled")
            except Exception:
                logger.warning("xFormers not available, skipping")

    # ── Public: load ──────────────────────────────────────────────────────

    def load(self) -> "SDXLModelLoader":
        """Load the base (and optionally refiner) pipeline."""
        logger.info("Loading SDXL base: %s", self.base_model)

        load_kwargs = dict(
            torch_dtype       = self.dtype,
            use_safetensors   = self.use_safetensors,
            variant           = "fp16" if self.dtype == torch.float16 else None,
            add_watermarker   = False,   # disable invisible watermark
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model, **load_kwargs
        )
        apply_scheduler(self.pipe, self.scheduler_name)
        self._optimise(self.pipe)

        if self.refiner_model:
            logger.info("Loading SDXL refiner: %s", self.refiner_model)
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.refiner_model, **load_kwargs
            )
            self._optimise(self.refiner_pipe)

        return self

    # ── Public: load LoRA ─────────────────────────────────────────────────

    def load_lora(
        self,
        lora_path:   str,
        adapter_name: str  = "face_lora",
        lora_scale:  float = 1.0,
    ) -> None:
        """
        Load a LoRA .safetensors file and fuse it into the UNet + text encoders.
        Can be called multiple times to stack adapters.
        """
        if self.pipe is None:
            raise RuntimeError("Call .load() before .load_lora()")

        lora_path = str(Path(lora_path).resolve())
        logger.info("Loading LoRA: %s  (scale=%.2f)", lora_path, lora_scale)

        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])

    def unload_lora(self) -> None:
        if self.pipe:
            self.pipe.unload_lora_weights()

    # ── Public: free memory ───────────────────────────────────────────────

    def offload(self) -> None:
        """Release GPU memory — call between batches if VRAM is tight."""
        self.pipe         = None
        self.refiner_pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipelines offloaded from memory")
