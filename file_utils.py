# file_utils.py

import subprocess
import tarfile
import os
import shutil

# Define the WEIGHTS_INFO dictionary at the top
WEIGHTS_INFO = {
    "GROUNDING_DINO_WEIGHTS_URL": "https://weights.replicate.delivery/default/grounding-dino/grounding-dino.tar",
    "HF-CACHE": "https://weights.replicate.delivery/default/grounding-dino/bert-base-uncased.tar",
    "SAM_VIT_B": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "WEIGHTS_CACHE_DIR": "/src/weights",
    "SAM_WEIGHTS_LOCAL_FILE": "/src/sam_weights/sam_vit_b_01ec64.pth",
    "HUGGINGFACE_CACHE_DIR": "/src/hf-cache/"
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
    grounding_dino_weights_dir = WEIGHTS_INFO["WEIGHTS_CACHE_DIR"]

    if not os.path.exists(grounding_dino_weights_dir):
        print(f"Downloading Grounding DINO weights to {grounding_dino_weights_dir}...")

        try:
            # Use wget to download the Grounding DINO weights
            subprocess.check_call(["wget", WEIGHTS_INFO["GROUNDING_DINO_WEIGHTS_URL"], "-O", "/src/tmp_file"])

            # Extract the tarball
            print("Extracting Grounding DINO weights...")
            tar = tarfile.open("/src/tmp_file")
            tar.extractall(path="/src/tmp")
            tar.close()

            # Move the extracted files to the target directory
            os.rename("/src/tmp", grounding_dino_weights_dir)
            os.remove("/src/tmp_file")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading or extracting Grounding DINO weights: {e}")
            raise
    else:
        print(f"Grounding DINO weights already present at {grounding_dino_weights_dir}, skipping download.")

def download_sam_weights():
    """Download SAM weights if they do not already exist."""
    SAM_WEIGHTS_LOCAL_FILE = WEIGHTS_INFO["SAM_WEIGHTS_LOCAL_FILE"]

    if not os.path.exists(SAM_WEIGHTS_LOCAL_FILE):
        print(f"Downloading SAM weights to {SAM_WEIGHTS_LOCAL_FILE}...")

        try:
            # Use wget to download the SAM weights
            subprocess.check_call(["wget", WEIGHTS_INFO["SAM_VIT_B"], "-O", "/src/tmp_file"])

            # Move the SAM weights to the target directory
            shutil.move("/src/tmp_file", SAM_WEIGHTS_LOCAL_FILE)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading SAM weights: {e}")
            raise
    else:
        print(f"SAM weights already present at {SAM_WEIGHTS_LOCAL_FILE}, skipping download.")
