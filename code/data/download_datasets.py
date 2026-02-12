import requests
import zipfile
import io
from pathlib import Path

ZENODO_URL = "https://zenodo.org/record/18621582/files/datasets.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"


def main():
    if DATASETS_DIR.exists():
        print("Datasets already present.")
        return

    print("Downloading datasets from Zenodo...")
    r = requests.get(ZENODO_URL)
    r.raise_for_status()

    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(PROJECT_ROOT)

    print("Done. Datasets ready.")


if __name__ == "__main__":
    main()
