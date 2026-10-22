
from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

import model


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path("data/LOLdataset/eval15/low")
REFERENCE_DIR = Path("data/LOLdataset/eval15/high")
OUTPUT_DIR = Path("LOLdataset_evaluation_results")

# Change these paths if your checkpoints have different names.
RUNS = {
    "Seed_1": Path("snapshots_seed_1/Epoch99.pth"),
    "Seed_2": Path("snapshots_seed_2/Epoch99.pth"),
    "Seed_3": Path("snapshots_seed_3/Epoch99.pth"),
}

# Must match the scale_factor used during training.
SCALE_FACTOR = 1

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Image utilities
# ============================================================

def get_images(folder):
    """Return images keyed by relative path, without extension."""
    return {
        str(path.relative_to(folder).with_suffix("")).casefold(): path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def load_image(path):
    """Load an image as RGB float32 in [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb_image(array, path):
    """Save an RGB array in [0, 1] as an image."""
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(
        (array * 255.0).round().astype(np.uint8)
    )
    image.save(path)


def save_before_after(low, pred, output_path):
    """
    Save a side-by-side comparison:
      left  = original low-light image
      right = enhanced image
    """
    before = Image.fromarray(
        (np.clip(low, 0, 1) * 255).round().astype(np.uint8)
    )
    after = Image.fromarray(
        (np.clip(pred, 0, 1) * 255).round().astype(np.uint8)
    )

    if before.size != after.size:
        raise ValueError(
            f"Image sizes differ: before={before.size}, "
            f"after={after.size}"
        )

    label_height = 36
    comparison = Image.new(
        "RGB",
        (before.width * 2, before.height + label_height),
        "white",
    )

    comparison.paste(before, (0, label_height))
    comparison.paste(after, (before.width, label_height))

    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), "Before (Low-light)", fill="black")
    draw.text(
        (before.width + 10, 10),
        "After (Enhanced)",
        fill="black",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)


# ============================================================
# Model inference
# ============================================================

def enhance_image(net, low):
    """Run Zero-DCE++ on one RGB image and return an RGB array."""
    tensor = torch.from_numpy(
        low.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        enhanced, _ = net(tensor)

    pred = enhanced.squeeze(0).cpu().numpy()
    pred = pred.transpose(1, 2, 0)

    return np.clip(pred, 0.0, 1.0)


def load_model(checkpoint_path):
    """Load a checkpoint into the matching Zero-DCE++ architecture."""
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
        # Compatibility with older PyTorch versions.
        state = torch.load(
            checkpoint_path,
            map_location=device,
        )

    net.load_state_dict(state)
    net.eval()
    return net


# ============================================================
# Evaluation
# ============================================================

def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test input folder not found: {TEST_DIR.resolve()}"
        )

    inputs = get_images(TEST_DIR)

    if not inputs:
        raise RuntimeError(
            f"No test images found in {TEST_DIR.resolve()}"
        )

    # Reference images are optional for image generation,
    # but required for PSNR, SSIM, and MAE.
    has_references = REFERENCE_DIR.is_dir()

    if has_references:
        references = get_images(REFERENCE_DIR)
        pairs = sorted(inputs.keys() & references.keys())

        if not pairs:
            print(
                "WARNING: No matching input/reference filenames. "
                "Will generate enhanced and comparison images only; "
                "full-reference metrics will not be calculated."
            )
            has_references = False
    else:
        references = {}
        pairs = []
        print(
            f"Reference folder not found: {REFERENCE_DIR.resolve()}\n"
            "Will generate enhanced and comparison images only."
        )

    # Without references, process every low-light image.
    image_keys = pairs if has_references else sorted(inputs.keys())

    if has_references and inputs.keys() != references.keys():
        print(
            f"Matched {len(pairs)} pairs from "
            f"{len(inputs)} inputs and {len(references)} references."
        )

    print(f"Test images to process: {len(image_keys)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_image_results = []
    run_summaries = []

    for run_name, checkpoint in RUNS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint.resolve()}"
            )

        print(f"\n{'=' * 60}")
        print(f"Run: {run_name}")
        print(f"Checkpoint: {checkpoint}")
        print(f"{'=' * 60}")

        net = load_model(checkpoint)

        run_dir = OUTPUT_DIR / run_name
        enhanced_dir = run_dir / "enhanced"
        comparison_dir = run_dir / "before_after"

        image_results = []

        for index, key in enumerate(image_keys, start=1):
            low_path = inputs[key]
            low = load_image(low_path)

            pred = enhance_image(net, low)

            relative_path = low_path.relative_to(TEST_DIR)

            # Save enhanced image.
            enhanced_path = (
                enhanced_dir / relative_path.with_suffix(".png")
            )
            save_rgb_image(pred, enhanced_path)

            # Save side-by-side before/after image.
            comparison_path = (
                comparison_dir / relative_path.with_suffix(".png")
            )
            save_before_after(low, pred, comparison_path)

            # Compute metrics only when a matching reference exists.
            if has_references:
                ref_path = references[key]
                ref = load_image(ref_path)

                if ref.shape != pred.shape:
                    raise ValueError(
                        f"Dimension mismatch for '{key}': "
                        f"prediction={pred.shape}, "
                        f"reference={ref.shape}. "
                        "Align the images before evaluation."
                    )

                psnr = peak_signal_noise_ratio(
                    ref, pred, data_range=1.0
                )
                ssim = structural_similarity(
                    ref,
                    pred,
                    channel_axis=-1,
                    data_range=1.0,
                )
                mae = float(np.mean(np.abs(ref - pred)))

                result = {
                    "seed": run_name,
                    "image": key,
                    "PSNR": float(psnr),
                    "SSIM": float(ssim),
                    "MAE": mae,
                }

                image_results.append(result)
                all_image_results.append(result)

            if index % 50 == 0 or index == len(image_keys):
                print(
                    f"Processed {index}/{len(image_keys)} images"
                )

        if image_results:
            run_summary = {
                "seed": run_name,
                "PSNR": float(np.mean([
                    row["PSNR"] for row in image_results
                ])),
                "SSIM": float(np.mean([
                    row["SSIM"] for row in image_results
                ])),
                "MAE": float(np.mean([
                    row["MAE"] for row in image_results
                ])),
            }

            run_summaries.append(run_summary)

            print(
                f"Mean PSNR: {run_summary['PSNR']:.4f} | "
                f"Mean SSIM: {run_summary['SSIM']:.4f} | "
                f"Mean MAE: {run_summary['MAE']:.6f}"
            )

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Export CSVs and print summary table
    # ========================================================

    if all_image_results:
        per_image_csv = OUTPUT_DIR / "metrics_per_image.csv"

        with per_image_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["seed", "image", "PSNR", "SSIM", "MAE"],
            )
            writer.writeheader()
            writer.writerows(all_image_results)

        summary_csv = OUTPUT_DIR / "metrics_summary.csv"

        with summary_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["seed", "PSNR", "SSIM", "MAE"],
            )
            writer.writeheader()
            writer.writerows(run_summaries)

        print("\n=== Metrics by seed (mean over test images) ===")
        print(
            f"{'Run':<14} {'PSNR ↑':>12} "
            f"{'SSIM ↑':>12} {'MAE ↓':>12}"
        )

        for row in run_summaries:
            print(
                f"{row['seed']:<14} "
                f"{row['PSNR']:>12.4f} "
                f"{row['SSIM']:>12.4f} "
                f"{row['MAE']:>12.6f}"
            )

        if len(run_summaries) >= 2:
            print("\n=== Mean ± sample standard deviation across runs ===")

            for metric in ("PSNR", "SSIM", "MAE"):
                values = [row[metric] for row in run_summaries]
                mean = np.mean(values)
                std = np.std(values, ddof=1)

                print(f"{metric}: {mean:.6f} ± {std:.6f}")

        print(f"\nPer-image metrics: {per_image_csv.resolve()}")
        print(f"Summary metrics:   {summary_csv.resolve()}")

    else:
        print(
            "\nNo full-reference metrics were calculated because "
            "paired normal-light reference images are unavailable."
        )

    print(f"\nAll output files: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
