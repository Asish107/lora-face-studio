"""
prompt_engine.py
----------------
Builds richly-detailed, identity-preserving prompts for SDXL + LoRA.

Key design goals:
  • Inject LoRA trigger word with configurable weight
  • Maximize realism / "human-made" feel via micro-imperfection vocabulary
  • Modular control over style / outfit / location / lighting / camera
  • Strong negative prompt bank
"""

from dataclasses import dataclass, field
from typing import Optional
import random


# ─────────────────────────────────────────────
# Style library
# ─────────────────────────────────────────────

STYLES: dict[str, dict] = {
    "hyperrealistic": {
        "positive": (
            "hyperrealistic portrait photography, shot on Sony A7R V, "
            "natural skin pores, micro hair detail, subsurface scattering, "
            "true-to-life color grading, unretouched editorial"
        ),
        "negative": (
            "painted, illustration, 3d render, cgi, plastic skin, "
            "over-smooth, airbrushed, filtered"
        ),
    },
    "cinematic": {
        "positive": (
            "cinematic portrait, anamorphic lens flare, film grain, "
            "Kodak Vision3 color palette, shallow depth of field, "
            "movie still, professional color grade"
        ),
        "negative": (
            "amateur photo, flat lighting, digital noise, phone camera, "
            "over-saturated, cartoon"
        ),
    },
    "fashion_photography": {
        "positive": (
            "high-fashion editorial photography, Vogue lighting, "
            "85mm f/1.4 prime lens, seamless backdrop or location, "
            "expert retouching that preserves natural skin texture, "
            "professional fashion photographer style"
        ),
        "negative": (
            "amateur, snapshot, harsh flash, blown highlights, "
            "unflattering angle, busy background"
        ),
    },
    "street_photography": {
        "positive": (
            "candid street photography, Leica M11, 35mm f/2 lens, "
            "natural ambient light, authentic unposed moment, "
            "urban grit, real-world imperfections, documentary feel"
        ),
        "negative": (
            "studio, posed, fake bokeh, HDR, artificial lighting, "
            "over-processed, stock photo look"
        ),
    },
    "charcoal": {
        "positive": (
            "charcoal portrait drawing, rough paper texture, "
            "loose expressive strokes, tonal value study, "
            "smudged mid-tones, hand-drawn imperfections, "
            "traditional artist's sketchbook, monochromatic"
        ),
        "negative": (
            "color, digital art, smooth, vector, cgi, photorealistic"
        ),
    },
    "watercolor": {
        "positive": (
            "loose watercolor portrait, wet-on-wet technique, "
            "visible brushstrokes, color blooms and bleeds, "
            "paper texture visible, painted edges, "
            "artist's color palette, traditional medium"
        ),
        "negative": (
            "digital, sharp edges, photorealistic, cgi, 3d render"
        ),
    },
    "anime": {
        "positive": (
            "studio Ghibli-inspired anime portrait, soft cel-shading, "
            "expressive hand-drawn eyes, clean linework, "
            "warm pastel palette, animated film quality"
        ),
        "negative": (
            "photorealistic, 3d render, deformed anatomy, "
            "nsfw, extra limbs"
        ),
    },
    "oil_painting": {
        "positive": (
            "classical oil painting portrait, visible brushwork, "
            "impasto texture, old masters lighting, "
            "linen canvas texture, glazing technique, "
            "rich color depth, museum quality"
        ),
        "negative": (
            "photo, digital, flat, airbrushed, cgi, modern"
        ),
    },
}


# ─────────────────────────────────────────────
# Lighting presets
# ─────────────────────────────────────────────

LIGHTING_PRESETS = {
    "golden_hour":    "warm golden hour backlight, rim lighting, lens flare, long shadows",
    "overcast":       "soft diffused overcast daylight, even shadows, natural skin tones",
    "studio_soft":    "large softbox key light at 45°, subtle fill card, catch-lights in eyes",
    "dramatic_side":  "hard single-source side lighting, deep shadow on half face, Rembrandt pattern",
    "window_natural": "soft directional window light, gentle fall-off, natural indoor ambience",
    "neon_night":     "urban neon reflections, cyan and magenta mixed light, night city glow",
    "campfire":       "warm flickering campfire light, orange-red cast, organic shadow movement",
    "foggy_morning":  "cool diffused morning fog, muted colour palette, atmospheric haze",
}


# ─────────────────────────────────────────────
# Micro-imperfection vocabulary (anti-AI feel)
# ─────────────────────────────────────────────

SKIN_REALISM_TOKENS = [
    "natural skin pores",
    "subtle skin texture",
    "slight asymmetry of features",
    "natural under-eye shadows",
    "real hair flyaways",
    "uneven lip line",
    "micro skin imperfections",
    "natural skin tone variation",
    "authentic eye moisture",
    "natural facial hair shadow",
]

CAMERA_REALISM_TOKENS = [
    "85mm portrait lens",
    "f/1.8 shallow depth of field",
    "slight chromatic aberration",
    "natural lens vignetting",
    "optical bokeh",
    "subtle film grain",
]


# ─────────────────────────────────────────────
# Negative prompt bank
# ─────────────────────────────────────────────

BASE_NEGATIVE = (
    # Identity & anatomy
    "face distortion, identity drift, different person, face swap, "
    "extra fingers, malformed hands, fused fingers, extra limbs, "
    "duplicate, cloned face, "
    # AI artifacts
    "cgi, 3d render, 3d model, ai artifacts, digital painting look, "
    "over-smooth skin, plastic skin, rubber skin, airbrushed, "
    "overly symmetrical face, uncanny valley, "
    # Technical flaws
    "blurry, out of focus background (when unintentional), "
    "overexposed, underexposed, jpeg artifacts, watermark, "
    "text, logo, cropped head, bad framing, "
    # Style pollution
    "cartoon (unless requested), anime (unless requested), "
    "illustration (unless requested), sketch (unless requested), "
    "low quality, low resolution, worst quality, bad anatomy"
)


# ─────────────────────────────────────────────
# Dataclass: PromptConfig
# ─────────────────────────────────────────────

@dataclass
class PromptConfig:
    trigger_word:    str
    lora_weight:     float         = 1.0
    outfit:          str           = "casual everyday clothing"
    location:        str           = "natural outdoor setting"
    lighting:        str           = "overcast"          # key from LIGHTING_PRESETS
    style:           str           = "hyperrealistic"    # key from STYLES
    camera_angle:    str           = "eye-level portrait"
    extra_positive:  str           = ""
    extra_negative:  str           = ""
    realism_tokens:  int           = 4   # how many micro-imperfection tokens to add
    seed:            Optional[int] = None


# ─────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────

class PromptEngine:

    def __init__(self, trigger_word: str, lora_weight: float = 1.0):
        self.trigger_word = trigger_word
        self.lora_weight  = lora_weight

    # ── Trigger injection ─────────────────────────────────────────────────

    def _trigger(self, weight: float) -> str:
        """
        Returns the AUTOMATIC1111-style LoRA tag:
            <lora:person_name:1.0>
        AND the inline trigger word for the text prompt.
        """
        return (
            f"{self.trigger_word}, "
            f"<lora:{self.trigger_word}:{weight:.2f}>"
        )

    # ── Realism vocabulary sampler ────────────────────────────────────────

    @staticmethod
    def _sample_realism(n: int = 4) -> str:
        tokens = random.sample(SKIN_REALISM_TOKENS, min(n, len(SKIN_REALISM_TOKENS)))
        cam    = random.sample(CAMERA_REALISM_TOKENS, 2)
        return ", ".join(tokens + cam)

    # ── Public: build positive prompt ────────────────────────────────────

    def build_positive(self, cfg: PromptConfig) -> str:
        style_block   = STYLES.get(cfg.style, STYLES["hyperrealistic"])
        lighting_desc = LIGHTING_PRESETS.get(cfg.lighting, cfg.lighting)
        realism_str   = self._sample_realism(cfg.realism_tokens)

        parts = [
            # 1. Identity anchor
            f"portrait of {self._trigger(cfg.lora_weight)}, "
            f"same person, consistent face, consistent identity",
            # 2. Scene
            f"{cfg.outfit}",
            f"{cfg.location}",
            f"{cfg.camera_angle}",
            # 3. Lighting
            lighting_desc,
            # 4. Style
            style_block["positive"],
            # 5. Realism micro-tokens
            realism_str,
            # 6. Quality anchors
            "highly detailed, sharp focus, masterwork, human-made",
        ]

        if cfg.extra_positive:
            parts.append(cfg.extra_positive)

        return ", ".join(p.strip().rstrip(",") for p in parts if p.strip())

    # ── Public: build negative prompt ────────────────────────────────────

    def build_negative(self, cfg: PromptConfig) -> str:
        style_block = STYLES.get(cfg.style, STYLES["hyperrealistic"])
        parts = [BASE_NEGATIVE, style_block.get("negative", "")]
        if cfg.extra_negative:
            parts.append(cfg.extra_negative)
        return ", ".join(p.strip().rstrip(",") for p in parts if p.strip())

    # ── Public: convenience all-in-one ───────────────────────────────────

    def build(self, cfg: PromptConfig) -> tuple[str, str]:
        """Returns (positive_prompt, negative_prompt)."""
        return self.build_positive(cfg), self.build_negative(cfg)


# ─────────────────────────────────────────────
# Quick demo / sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engine = PromptEngine(trigger_word="jhndoe", lora_weight=0.9)

    cfg = PromptConfig(
        trigger_word  = "jhndoe",
        outfit        = "navy blue linen blazer, open collar white shirt",
        location      = "Parisian café terrace at dusk",
        lighting      = "golden_hour",
        style         = "cinematic",
        camera_angle  = "three-quarter view, slight upward angle",
    )

    pos, neg = engine.build(cfg)

    print("── POSITIVE ─────────────────────────────────")
    print(pos)
    print("\n── NEGATIVE ─────────────────────────────────")
    print(neg)
