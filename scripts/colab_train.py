"""
scripts/colab_train.py
-----------------------
Self-contained Colab bootstrap script.

In Colab:  !python scripts/colab_train.py --trigger YOUR_NAME --drive_folder MyFaceLoRA

What it does:
  1. Installs all dependencies
  2. Mounts Google Drive (optional)
  3. Clones kohya-ss sd-scripts
  4. Runs dataset_prep then train_lora
"""

import os
import sys
import subprocess
import argparse


def run(cmd: str) -> None:
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def install_deps() -> None:
    print("\n── Installing dependencies ──────────────────────────────────")
    packages = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "diffusers[torch] transformers accelerate",
        "bitsandbytes xformers",
        "Pillow opencv-python-headless",
        "safetensors pyyaml",
        "fastapi uvicorn[standard] python-multipart",
        "tensorboard",
    ]
    for pkg in packages:
        run(f"{sys.executable} -m pip install -q {pkg}")

    # kohya-ss sd-scripts
    if not os.path.exists("sd-scripts"):
        run("git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git")
        # Fix the problematic local reference in sd-scripts requirements
        run(r"sed -i 's/^\.\///g' sd-scripts/requirements.txt || true")
        run(r"sed -i '/^-e \./d' sd-scripts/requirements.txt || true")
        run(f"{sys.executable} -m pip install -q -r sd-scripts/requirements.txt")


def mount_drive(drive_folder: str) -> str:
    """Mount Google Drive and return the upload folder path."""
    drive_path = "/content/drive"
    if os.path.exists(drive_path):
        print("✓ Google Drive already mounted.")
    else:
        try:
            from google.colab import drive
            drive.mount(drive_path)
        except Exception as e:
            print(f"Warning: Could not mount drive automatically: {e}")
            print("Please mount Drive manually using the folder icon on the left.")

    folder = f"/content/drive/MyDrive/{drive_folder}"
    os.makedirs(folder, exist_ok=True)
    print(f"Working folder: {folder}")
    return folder


def upload_images_colab(dest: str) -> None:
    """Colab file-picker widget to upload training images."""
    try:
        from google.colab import files
        print(f"\nUpload your face images (15–25 photos):")
        uploaded = files.upload()
        os.makedirs(dest, exist_ok=True)
        for name, data in uploaded.items():
            with open(os.path.join(dest, name), "wb") as f:
                f.write(data)
        print(f"✓ {len(uploaded)} images saved to {dest}")
    except ImportError:
        print("Not in Colab — place images manually in the source folder")


def main():
    parser = argparse.ArgumentParser(description="Colab LoRA training bootstrap")
    parser.add_argument("--trigger",       required=True,  help="LoRA trigger word")
    parser.add_argument("--name",          default=None,   help="Person name (default = trigger)")
    parser.add_argument("--drive_folder",  default="LoRaFaceStudio")
    parser.add_argument("--epochs",        type=int, default=25)
    parser.add_argument("--dim",           type=int, default=32)
    parser.add_argument("--skip_install",  action="store_true")
    args = parser.parse_args()

    person_name = args.name or args.trigger

    if not args.skip_install:
        install_deps()

    # ── Drive + paths ─────────────────────────────────────────────────────
    base_dir   = mount_drive(args.drive_folder)
    raw_dir    = os.path.join(base_dir, "raw_images")
    data_dir   = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "output")

    os.makedirs(raw_dir,    exist_ok=True)
    os.makedirs(data_dir,   exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Upload images ────────────────────────────────────────────────────
    # upload_images_colab(raw_dir)
    # Note: Automated upload picker fails in script mode. 
    # Please upload your images to: Google Drive > LoRaFaceStudio > raw_images

    # ── Dataset prep ─────────────────────────────────────────────────────
    run(
        f"{sys.executable} train/dataset_prep.py "
        f"--source {raw_dir} "
        f"--output {data_dir} "
        f"--trigger {args.trigger} "
        f"--name {person_name} "
        f"--no-caption "
        f"--no-crop"    # If face detection fails, use the whole image instead of skipping
    )

    # ── Heartbeat Check ──────────────────────────────────────────────────
    print("\n── Library Heartbeat Check ──────────────────────────────────")
    run(f"export TF_CPP_MIN_LOG_LEVEL=3 && export PYTHONPATH={os.path.join(os.getcwd(), 'sd-scripts')}:$PYTHONPATH && {sys.executable} sd-scripts/sdxl_train_network.py --help | head -n 1")

    # ── Training ─────────────────────────────────────────────────────────
    run(
        f"{sys.executable} train/train_lora.py "
        f"--trigger {args.trigger} "
        f"--data_dir {data_dir} "
        f"--output {output_dir} "
        f"--epochs {args.epochs} "
        f"--dim {args.dim} "
        f"--no-accelerate"
    )

    print(f"\n✓  All done! LoRA saved to: {output_dir}")
    print(f"   Trigger word: '{args.trigger}'")
    print(f"\n   To generate images:")
    print(f"   python inference/generate.py --lora {output_dir}/{args.trigger}_lora.safetensors \\")
    print(f"       --trigger {args.trigger} --style cinematic --batch")


if __name__ == "__main__":
    main()
