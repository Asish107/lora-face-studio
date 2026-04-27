"""
dataset_prep.py
---------------
Prepares a face dataset for SDXL LoRA training.
Pipeline: raw images → face-crop → resize → caption → kohya-ss folder structure
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

# Optional — gracefully skipped if not installed
try:
    import cv2
    import numpy as np
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    _BLIP = True
except ImportError:
    _BLIP = False


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

TARGET_SIZE   = 1024          # SDXL native resolution
MIN_FACE_SIZE = 200           # px — discard tiny face crops
REPEATS       = 10            # kohya repeat count in folder name


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_and_pad(img: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """
    Resize longest side to `size`, then pad the shorter side
    with mirrored edges so no content is cropped.
    """
    w, h   = img.size
    ratio  = size / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    img    = img.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    offset = ((size - nw) // 2, (size - nh) // 2)
    canvas.paste(img, offset)
    return canvas


def crop_face_opencv(img: Image.Image, padding: float = 0.40) -> Optional[Image.Image]:
    """
    Detect and crop the largest face using OpenCV Haar cascades.
    Returns None if no face is found or cv2 is unavailable.
    """
    if not _CV2:
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade      = cv2.CascadeClassifier(cascade_path)
    arr          = np.array(img)
    gray         = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    if len(faces) == 0:
        return None

    # Take the largest detection
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.width,  x + w + pad_x)
    y2 = min(img.height, y + h + pad_y)
    return img.crop((x1, y1, x2, y2))


# ─────────────────────────────────────────────
# Caption generation
# ─────────────────────────────────────────────

def generate_caption_blip(img: Image.Image, trigger: str) -> str:
    """
    Use BLIP to auto-caption an image, then prepend the LoRA trigger word.
    Falls back to a template caption if BLIP is unavailable.
    """
    if not _BLIP:
        return (
            f"photo of {trigger}, "
            "a person with natural skin texture, realistic portrait, "
            "subtle facial asymmetry, high detail"
        )

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    inputs  = processor(img, return_tensors="pt").to(device)
    out     = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Strip generic "a person" openers — we inject the trigger word instead
    caption = re.sub(r"^(a photo of |a picture of |a )?a person ", "", caption)
    return f"photo of {trigger}, {caption}, natural skin texture, realistic imperfections"


def write_caption(caption: str, img_path: str) -> None:
    txt_path = Path(img_path).with_suffix(".txt")
    txt_path.write_text(caption, encoding="utf-8")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def prepare_dataset(
    source_dir:     str,
    output_dir:     str,
    trigger_word:   str,
    person_name:    str,
    crop_faces:     bool = True,
    auto_caption:   bool = True,
    repeats:        int  = REPEATS,
) -> None:
    """
    Full pipeline:
      1. Read raw images from source_dir
      2. (Optional) Detect & crop faces
      3. Resize + pad to 1024×1024
      4. Generate/write captions
      5. Save into kohya-ss folder structure:
         output_dir/<repeats>_<person_name>/
    """
    source = Path(source_dir)
    exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in source.iterdir() if p.suffix.lower() in exts]

    if not images:
        raise ValueError(f"No images found in {source_dir}")

    # kohya-ss expects:  <n_repeats>_<concept_name>
    concept_folder = Path(output_dir) / f"{repeats}_{person_name}"
    concept_folder.mkdir(parents=True, exist_ok=True)

    print(f"[dataset_prep] Processing {len(images)} images → {concept_folder}")

    kept = 0
    for idx, img_path in enumerate(sorted(images)):
        img = load_image(str(img_path))

        if crop_faces:
            cropped = crop_face_opencv(img)
            if cropped is None:
                print(f"  [skip] No face detected: {img_path.name}")
                continue
            img = cropped

        img = resize_and_pad(img)

        out_name = f"{person_name}_{idx:04d}.png"
        out_path = concept_folder / out_name
        img.save(out_path, format="PNG")

        if auto_caption:
            caption = generate_caption_blip(img, trigger_word)
        else:
            caption = (
                f"photo of {trigger_word}, "
                "natural skin texture, subtle facial asymmetry, "
                "high detail portrait, realistic imperfections"
            )

        write_caption(caption, str(out_path))
        kept += 1
        print(f"  [{idx+1}/{len(images)}] saved {out_name}")

    print(f"\n[dataset_prep] ✓  {kept} images ready in: {concept_folder}")
    print(f"[dataset_prep]    Trigger word: '{trigger_word}'")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LoRA face dataset")
    parser.add_argument("--source",      required=True,  help="Folder with raw images")
    parser.add_argument("--output",      required=True,  help="Output dataset root folder")
    parser.add_argument("--trigger",     required=True,  help="LoRA trigger word (e.g. jhndoe)")
    parser.add_argument("--name",        required=True,  help="Person/concept name")
    parser.add_argument("--no-crop",     action="store_true", help="Skip face cropping")
    parser.add_argument("--no-caption",  action="store_true", help="Use template captions only")
    parser.add_argument("--repeats",     type=int, default=REPEATS)
    args = parser.parse_args()

    prepare_dataset(
        source_dir   = args.source,
        output_dir   = args.output,
        trigger_word = args.trigger,
        person_name  = args.name,
        crop_faces   = not args.no_crop,
        auto_caption = not args.no_caption,
        repeats      = args.repeats,
    )
