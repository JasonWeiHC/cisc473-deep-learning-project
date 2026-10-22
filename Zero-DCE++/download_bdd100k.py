
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import shutil

REPO_ID = "dgural/bdd100k"
OUTPUT_DIR = Path("data/bdd100k_subset")
NUM_IMAGES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

# List files in the dataset's data folder
files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

# Select 100 JPG images
images = sorted(
    f for f in files
    if f.startswith("data/")
    and f.lower().endswith((".jpg", ".jpeg"))
)[:NUM_IMAGES]

if len(images) < NUM_IMAGES:
    raise RuntimeError(f"Only found {len(images)} images")

print(f"Downloading {len(images)} images...")

for i, filename in enumerate(images, start=1):
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
    )

    destination = OUTPUT_DIR / Path(filename).name
    shutil.copy2(downloaded_path, destination)

    print(f"[{i}/{len(images)}] {destination.name}")

print(f"\nDone! Images saved to: {OUTPUT_DIR.resolve()}")
