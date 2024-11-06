import subprocess
import tarfile
import os

# Download and extract weights, then move them to the "dest" directory.
def download_weights(url, dest):
    if not os.path.exists("/src/tmp.tar"):
        print(f"Downloading {url}...")
        try:
            output = subprocess.check_output(["pget", url, "/src/tmp.tar"])
        except subprocess.CalledProcessError as e:
            raise e
    # Extract and move weights
    tar = tarfile.open("/src/tmp.tar")
    tar.extractall(path="/src/tmp")
    tar.close()
    os.rename("/src/tmp", dest)
    os.remove("/src/tmp.tar")

# Define URLs for weights
WEIGHTS_URL_DIR_MAP = {
    "GROUNDING_DINO_WEIGHTS_URL": "https://weights.replicate.delivery/default/grounding-dino/grounding-dino.tar",
    "HF-CACHE": "https://weights.replicate.delivery/default/grounding-dino/bert-base-uncased.tar",
    "SAM2_TINY_CHECKPOINT": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
}

# Download functions
def download_grounding_dino_weights(grounding_dino_weights_dir, hf_cache_dir):
    """Download weights for GroundingDINO."""
    if not os.path.exists(grounding_dino_weights_dir):
        download_weights(WEIGHTS_URL_DIR_MAP["GROUNDING_DINO_WEIGHTS_URL"], grounding_dino_weights_dir)
    if not os.path.exists(hf_cache_dir):
        download_weights(WEIGHTS_URL_DIR_MAP["HF-CACHE"], hf_cache_dir)

def download_sam2_tiny_weights(sam2_weights_dir):
    """Download weights for SAM-2 tiny model."""
    if not os.path.exists(sam2_weights_dir):
        download_weights(WEIGHTS_URL_DIR_MAP["SAM2_TINY_CHECKPOINT"], sam2_weights_dir)
