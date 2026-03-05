"""
download_weights.py
-------------------
Downloads model weights from Hugging Face Hub at container startup
if they are not already present locally.

Usage: Set HF_REPO_ID env variable to your HF repo, e.g.:
  export HF_REPO_ID="your-username/osteoporosis-knee-xray"
  python download_weights.py
"""
import os
import urllib.request

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Set this to your Hugging Face repo ID ──────────────────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
WEIGHTS = ["resnet_best.pth", "densenet_best.pth", "effnet_best.pth"]


def download_from_hf(repo_id, filename, dest_path):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"Downloading {filename} from {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"  ✅ Saved {filename} ({size_mb:.1f} MB)")


def ensure_weights():
    if not HF_REPO_ID:
        print("HF_REPO_ID not set — assuming weights are baked into the image.")
        return

    for fname in WEIGHTS:
        dest = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(dest):
            print(f"  ✅ {fname} already present ({os.path.getsize(dest)/1e6:.1f} MB)")
        else:
            download_from_hf(HF_REPO_ID, fname, dest)


if __name__ == "__main__":
    ensure_weights()
    print("\nAll weights ready.")
