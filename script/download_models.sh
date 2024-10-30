#!/bin/bash

# Set up the weights directory
mkdir -p weights
cd weights

# Download the SAM model weight using wget
echo "Downloading SAM model weights..."
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth

# Clone GroundingDINO repository for code dependencies
echo "Cloning GroundingDINO repository..."
git clone https://github.com/IDEA-Research/GroundingDINO

# Install GroundingDINO as an editable Python package
echo "Installing GroundingDINO as a Python package..."
pip install -e GroundingDINO

# Clone Segment Anything repository for code dependencies
echo "Cloning Segment Anything repository..."
git clone https://github.com/facebookresearch/segment-anything

# Download GroundingDINO model weights from Hugging Face using Git LFS
echo "Installing Git LFS and downloading GroundingDINO weights..."
git lfs install
git clone https://huggingface.co/ShilongLiu/GroundingDINO

# Create hf_path_exports.py with paths to the GroundingDINO configuration and weight files
echo "Generating hf_path_exports.py with model paths..."
cat <<EOL >hf_path_exports.py
cache_config_file = "/src/weights/GroundingDINO/GroundingDINO_SwinB.cfg.py"
cache_file = "/src/weights/GroundingDINO/groundingdino_swinb_cogcoor.pth"
EOL

echo "All model weights and files have been successfully downloaded and installed in their respective directories."
