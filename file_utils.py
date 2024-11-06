# file_utils.py

import subprocess
import tarfile
import os

def download_weights(url, dest):
    if not os.path.exists("/src/tmp.tar"):
        print(f"Downloading {url}...")
        try:
            output = subprocess.check_output(["pget", url, "/src/tmp.tar"])
        except subprocess.CalledProcessError as e:
            raise e
    tar = tarfile.open("/src/tmp.tar")
    tar.extractall(path="/src/tmp")
    tar.close()
    os.rename("/src/tmp", dest)
    os.remove("/src/tmp.tar")

WEIGHTS_URL_DIR_MAP = {
    "GROUNDING_DINO_WEIGHTS_URL": "https://weights.replicate.delivery/default/grounding-dino/grounding-dino.tar",
    "HF-CACHE": "https://weights.replicate.delivery/default/grounding-dino/bert-base-uncased.tar",
    "SAM_VIT_B": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

def download_grounding_dino_weights(grounding_dino_weights_dir, hf_cache_dir):
    if not os.path.exists(grounding_dino_weights_dir):
        download_weights(WEIGHTS_URL_DIR_MAP["GROUNDING_DINO_WEIGHTS_URL"], grounding_dino_weights_dir)
    if not os.path.exists(hf_cache_dir):
        download_weights(WEIGHTS_URL_DIR_MAP["HF-CACHE"], hf_cache_dir)

def download_sam_weights(sam_weights_dir):
    if not os.path.exists(sam_weights_dir):
        download_weights(WEIGHTS_URL_DIR_MAP["SAM_VIT_B"], sam_weights_dir)
