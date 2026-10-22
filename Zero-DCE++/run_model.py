
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = Path("snapshots_Zero_DCE++/Epoch99.pth")

# Folder containing images to run model on
INPUT_DIR = Path("data/bdd100k_subset")

OUTPUT_DIR = Path("bdd100k_zero_dce_results")

# Must match the architecture used to train this checkpoint.
SCALE_FACTOR = 1

# Set to None to process every image, or choose a number for a test.
MAX_IMAGES = 100

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Utilities
# ============================================================

def load_rgb(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(
        (array * 255.0).round().astype(np.uint8)
    )
    image.save(path)


def save_comparison(before, after, path):
    """Save original and enhanced images side by side."""
    before_img = Image.fromarray(
        (np.clip(before, 0, 1) * 255).round().astype(np.uint8)
    )
    after_img = Image.fromarray(
        (np.clip(after, 0, 1) * 255).round().astype(np.uint8)
    )

    if before_img.size != after_img.size:
        raise ValueError(
            f"Dimension mismatch: {before_img.size} vs "
            f"{after_img.size}"
        )

    label_height = 36
    width, height = before_img.size

    canvas = Image.new(
        "RGB", (width * 2, height + label_height), "white"
    )
    canvas.paste(before_img, (0, label_height))
    canvas.paste(after_img, (width, label_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Before: BDD100K", fill="black")
    draw.text((width + 10, 10), "After: Zero-DCE++", fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def load_model(checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    net = model.enhance_net_nopool(
        scale_factor=SCALE_FACTOR
    ).to(device)

    try:
        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    net.load_state_dict(state)
    net.eval()
    return net


def enhance(net, image):
    tensor = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        enhanced, _ = net(tensor)

    result = enhanced.squeeze(0).cpu().numpy()
    return np.clip(result.transpose(1, 2, 0), 0.0, 1.0)


# ============================================================
# Main
# ============================================================

def main():
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT.resolve()}")
    print(f"Input directory: {INPUT_DIR.resolve()}")

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"BDD100K image directory not found:\n"
            f"{INPUT_DIR.resolve()}\n\n"
            "Update INPUT_DIR near the top of this script."
        )

    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {INPUT_DIR.resolve()}"
        )

    print(f"Images to process: {len(image_paths)}")

    net = load_model(CHECKPOINT)

    enhanced_dir = OUTPUT_DIR / "enhanced"
    comparison_dir = OUTPUT_DIR / "before_after"

    enhanced_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(INPUT_DIR)

        try:
            original = load_rgb(image_path)
            enhanced = enhance(net, original)

            output_name = relative_path.with_suffix(".png")

            save_rgb(
                enhanced,
                enhanced_dir / output_name,
            )

            save_comparison(
                original,
                enhanced,
                comparison_dir / output_name,
            )

        except Exception as exc:
            print(f"\nFailed on {image_path}: {exc}")
            raise

        if index % 10 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    print("\nFinished!")
    print(f"Enhanced images: {enhanced_dir.resolve()}")
    print(f"Before/after pairs: {comparison_dir.resolve()}")


if __name__ == "__main__":
    main()
