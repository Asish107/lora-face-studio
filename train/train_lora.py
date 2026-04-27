"""
train_lora.py
-------------
Launcher that:
  1. Validates the dataset structure
  2. Patches config.yaml with runtime values (trigger word, paths)
  3. Invokes kohya-ss sdxl_train_network.py
  4. Works locally (Mac/Linux) or in Google Colab

Usage:
    python train/train_lora.py \
        --trigger   jhndoe \
        --data_dir  ./data/dataset \
        --output    ./output/lora \
        --config    ./train/config.yaml
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

import yaml


# ─────────────────────────────────────────────
# Kohya-ss detection
# ─────────────────────────────────────────────

def find_kohya_script() -> Path:
    """Locate sdxl_train_network.py in common install paths."""
    candidates = [
        Path("./sd-scripts/sdxl_train_network.py"),
        Path(os.path.expanduser("~/sd-scripts/sdxl_train_network.py")),
        Path("/content/sd-scripts/sdxl_train_network.py"),   # Colab default
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def install_kohya_if_missing() -> Path:
    """Clone kohya-ss sd-scripts if not present (Colab / first-run)."""
    script = find_kohya_script()
    if script:
        return script

    print("[train] Cloning kohya-ss sd-scripts …")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/kohya-ss/sd-scripts.git"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         "sd-scripts/requirements.txt"],
        check=True,
    )
    script = find_kohya_script()
    if not script:
        raise RuntimeError("Could not locate sdxl_train_network.py after install.")
    return script


# ─────────────────────────────────────────────
# Config patching
# ─────────────────────────────────────────────

def patch_config(config_path: str, overrides: dict) -> str:
    """
    Load config.yaml, apply runtime overrides, write a temp copy,
    return the path to the patched file.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    patched_path = Path(config_path).with_suffix(".patched.yaml")
    with open(patched_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return str(patched_path)


# ─────────────────────────────────────────────
# Dataset validation
# ─────────────────────────────────────────────

def validate_dataset(data_dir: str, trigger: str) -> None:
    root = Path(data_dir)
    concept_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not concept_dirs:
        raise ValueError(
            f"No concept subdirectory found in {data_dir}.\n"
            "Run  python train/dataset_prep.py  first."
        )

    images = []
    for d in concept_dirs:
        images += list(d.glob("*.png")) + list(d.glob("*.jpg"))

    if len(images) < 3:
        raise ValueError(
            f"Only {len(images)} images found — need at least 3 for stable training."
        )

    # Check captions exist
    missing_captions = [
        p for p in images if not p.with_suffix(".txt").exists()
    ]
    if missing_captions:
        print(f"[warn] {len(missing_captions)} images missing caption files.")

    print(f"[validate] ✓  {len(images)} images across {len(concept_dirs)} concept folder(s)")


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def run_training(
    script:     Path,
    config:     str,
    accelerate: bool = True,
) -> None:
    if accelerate:
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "2",
            str(script),
            "--config_file", config,
        ]
    else:
        # Fallback for Mac MPS or CPU-only
        cmd = [sys.executable, str(script), "--config_file", config]

    print(f"\n[train] Launching:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")


# ─────────────────────────────────────────────
# Post-training summary
# ─────────────────────────────────────────────

def print_summary(output_dir: str, trigger: str) -> None:
    output = Path(output_dir)
    loras  = sorted(output.glob("*.safetensors"))
    print("\n" + "─" * 60)
    print("✓  Training complete!")
    print(f"   Trigger word : '{trigger}'")
    print(f"   Output dir   : {output_dir}")
    if loras:
        print("   LoRA files   :")
        for f in loras:
            size_mb = f.stat().st_size / 1_048_576
            print(f"     • {f.name}  ({size_mb:.1f} MB)")
    print("─" * 60)
    print("\nNext step — generate images:")
    print(f"  python inference/generate.py --lora {loras[-1] if loras else '<lora.safetensors>'} --trigger {trigger}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SDXL LoRA for face identity")
    parser.add_argument("--trigger",   required=True,  help="LoRA trigger word")
    parser.add_argument("--data_dir",  default="./data/dataset")
    parser.add_argument("--output",    default="./output/lora")
    parser.add_argument("--config",    default="./train/config.yaml")
    parser.add_argument("--model",     default=None,   help="Override base model path/HF id")
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--dim",       type=int, default=None, help="LoRA rank (network_dim)")
    parser.add_argument("--no-accelerate", action="store_true",
                        help="Use plain Python launch (Mac MPS / CPU)")
    args = parser.parse_args()

    # ── Step 1: validate dataset ──────────────────────────────────────────
    validate_dataset(args.data_dir, args.trigger)

    # ── Step 2: ensure kohya is available ────────────────────────────────
    script = install_kohya_if_missing()
    print(f"[train] Using kohya script: {script}")

    # ── Step 3: patch config ─────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    overrides = {
        "train_data_dir": args.data_dir,
        "output_dir":     args.output,
        "output_name":    f"{args.trigger}_lora",
    }
    if args.model:   overrides["pretrained_model_name_or_path"] = args.model
    if args.epochs:  overrides["max_train_epochs"] = args.epochs
    if args.dim:
        overrides["network_dim"]   = args.dim
        overrides["network_alpha"] = args.dim // 2

    patched_config = patch_config(args.config, overrides)
    print(f"[train] Config  : {patched_config}")

    # ── Step 4: run training ─────────────────────────────────────────────
    run_training(script, patched_config, accelerate=not args.no_accelerate)

    # ── Step 5: cleanup + summary ────────────────────────────────────────
    Path(patched_config).unlink(missing_ok=True)
    print_summary(args.output, args.trigger)


if __name__ == "__main__":
    main()
