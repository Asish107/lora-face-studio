"""
utils/image_utils.py
--------------------
Image processing helpers used across training and inference.
"""

import io
import math
from pathlib import Path
from typing import Union

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


# ─────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────
ImageLike = Union[str, Path, Image.Image]


def load(src: ImageLike) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    return Image.open(str(src)).convert("RGB")


# ─────────────────────────────────────────────
# Resizing / cropping
# ─────────────────────────────────────────────

def resize_to_square(img: Image.Image, size: int, fill: tuple = (127,127,127)) -> Image.Image:
    """Resize longest side to `size`, pad shorter side symmetrically."""
    w, h  = img.size
    ratio = size / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    img    = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Hard center crop to `size × size`."""
    w, h = img.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def upscale(img: Image.Image, factor: int = 2) -> Image.Image:
    """Simple Lanczos upscale."""
    return img.resize((img.width * factor, img.height * factor), Image.LANCZOS)


# ─────────────────────────────────────────────
# Post-processing — make output look less "AI"
# ─────────────────────────────────────────────

def add_film_grain(img: Image.Image, strength: float = 0.03) -> Image.Image:
    """
    Overlay subtle luminance grain that mimics high-ISO film.
    strength: 0.01 (barely visible) → 0.08 (noticeable grain)
    """
    arr  = np.array(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0, strength, arr.shape[:2])
    # Apply grain only to luminance channel via broadcasting
    arr[:, :, 0] = np.clip(arr[:, :, 0] + noise, 0, 1)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + noise, 0, 1)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def add_vignette(img: Image.Image, strength: float = 0.25) -> Image.Image:
    """
    Subtle lens vignette darkening towards corners.
    strength: 0 (none) → 1 (very dark corners)
    """
    w, h   = img.size
    arr    = np.array(img, dtype=np.float32)
    cx, cy = w / 2, h / 2

    # Gaussian falloff from centre
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = 1 - np.clip(dist * strength, 0, strength)
    mask = np.stack([mask] * 3, axis=-1)

    arr = np.clip(arr * mask, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def subtle_sharpening(img: Image.Image, radius: float = 1.0, amount: float = 0.4) -> Image.Image:
    """
    Unsharp-mask sharpening that brings out skin and hair micro-detail
    without the plasticky over-sharpen look.
    """
    from PIL import ImageFilter
    blurred   = img.filter(ImageFilter.GaussianBlur(radius))
    arr_orig  = np.array(img,     dtype=np.float32)
    arr_blur  = np.array(blurred, dtype=np.float32)
    arr_sharp = np.clip(arr_orig + amount * (arr_orig - arr_blur), 0, 255)
    return Image.fromarray(arr_sharp.astype(np.uint8))


def colour_grade(img: Image.Image, contrast: float = 1.05, saturation: float = 0.95) -> Image.Image:
    """
    Very mild colour grade:
      - Tiny contrast boost → more pop without crushing
      - Slight desaturation → closer to film / editorial look
    """
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def apply_realism_postprocess(img: Image.Image) -> Image.Image:
    """
    Full realism pipeline applied in the right order:
      sharpen → grain → vignette → colour grade
    Call this on every generated image before saving.
    """
    img = subtle_sharpening(img, radius=0.8, amount=0.35)
    img = add_film_grain(img, strength=0.018)
    img = add_vignette(img, strength=0.18)
    img = colour_grade(img, contrast=1.04, saturation=0.93)
    return img


# ─────────────────────────────────────────────
# Grid builder
# ─────────────────────────────────────────────

def make_grid(
    images:  list[Image.Image],
    cols:    int = 0,
    padding: int = 4,
    bg:      tuple = (15, 15, 15),
) -> Image.Image:
    """Arrange a list of images into a grid with a dark background."""
    if not images:
        raise ValueError("Empty image list")

    cols  = cols or math.ceil(math.sqrt(len(images)))
    rows  = math.ceil(len(images) / cols)
    w, h  = images[0].size
    gw    = cols * w + (cols + 1) * padding
    gh    = rows * h + (rows + 1) * padding
    grid  = Image.new("RGB", (gw, gh), bg)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x    = padding + c * (w + padding)
        y    = padding + r * (h + padding)
        grid.paste(img.resize((w, h)), (x, y))

    return grid


# ─────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────

def to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
