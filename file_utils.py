# file_utils.py

import subprocess
import tarfile
import os
import shutil

# Define the WEIGHTS_INFO dictionary at the top
WEIGHTS_INFO = {
    "DINO_WEIGHTS_URL": "https://weights.replicate.delivery/default/grounding-dino/grounding-dino.tar",
    "HF-CACHE": "https://weights.replicate.delivery/default/grounding-dino/bert-base-uncased.tar",
    "SAM_VIT_B": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "SAM_VIT_L": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "SAM_VIT_H": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "DINO_WEIGHTS_DIR": "/src/weights_dino",
    "SAM_WEIGHTS_DIR": "/src/weights_sam",
    "SAM_WEIGHTS_LOCAL_FILE": "/src/weights_sam/sam_vit_l_0b3195.pth",
    "HUGGINGFACE_CACHE_DIR": "/src/hf-cache/",
    "TEMP_DIR": "/src/tmp"
}

def download_weights():
    """Download all necessary weights for the models."""
    download_grounding_dino_weights()
    download_sam_weights()
    download_bert_base_uncased() 

def download_bert_base_uncased():
    """Download the bert-base-uncased model and tokenizer to the cache directory."""
    # Import transformers here to avoid importing it at the top level
    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    cache_dir = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]
    print(f"Downloading {model_name} model and tokenizer to {cache_dir}...")
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

def download_grounding_dino_weights():
    """Download and extract Grounding DINO weights if they do not already exist."""
    DINO_WEIGHTS_DIR = WEIGHTS_INFO["DINO_WEIGHTS_DIR"]
    TEMP_DIR = WEIGHTS_INFO["TEMP_DIR"]
    temp_file = os.path.join(TEMP_DIR, "tmp_file")

    # Ensure directories exist
    os.makedirs(DINO_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Check if DINO weights directory contains any files
    if not os.listdir(DINO_WEIGHTS_DIR):  # Proceed if directory is empty
        print(f"Downloading Grounding DINO weights to {DINO_WEIGHTS_DIR}...")

        try:
            # Use wget to download the Grounding DINO weights
            subprocess.check_call(["wget", WEIGHTS_INFO["DINO_WEIGHTS_URL"], "-O", temp_file])

            # Extract the tarball
            print("Extracting Grounding DINO weights...")
            with tarfile.open(temp_file) as tar:
                tar.extractall(path=DINO_WEIGHTS_DIR)

            # Clean up temporary file
            os.remove(temp_file)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading or extracting Grounding DINO weights: {e}")
            raise
    else:
        print(f"Grounding DINO weights already present at {DINO_WEIGHTS_DIR}, skipping download.")

def download_sam_weights():
    """Download SAM weights if they do not already exist."""
    SAM_WEIGHTS_LOCAL_FILE = WEIGHTS_INFO["SAM_WEIGHTS_LOCAL_FILE"]
    SAM_WEIGHTS_DIR = WEIGHTS_INFO["SAM_WEIGHTS_DIR"]
    TEMP_DIR = WEIGHTS_INFO["TEMP_DIR"]
    temp_file = os.path.join(TEMP_DIR, "tmp_file")

    # Ensure both TEMP_DIR and SAM_WEIGHTS_DIR exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SAM_WEIGHTS_DIR, exist_ok=True)

    if not os.path.exists(SAM_WEIGHTS_LOCAL_FILE):
        print(f"Downloading SAM weights to {SAM_WEIGHTS_LOCAL_FILE}...")

        try:
            # Use wget to download the SAM weights
            subprocess.check_call(["wget", WEIGHTS_INFO["SAM_VIT_L"], "-O", temp_file])

            # Move the SAM weights to the target directory
            shutil.move(temp_file, SAM_WEIGHTS_LOCAL_FILE)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading SAM weights: {e}")
            raise
    else:
        print(f"SAM weights already present at {SAM_WEIGHTS_LOCAL_FILE}, skipping download.")

