# predict.py

import os
import uuid
from file_utils import WEIGHTS_INFO

# Set environment variables before any other imports
os.environ["HF_HOME"] = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]
os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]
os.environ["TRANSFORMERS_CACHE"] = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]

# Now import the rest
from cog import BasePredictor, Input, Path
from typing import List
import torch
from segment_anything import SamPredictor, sam_model_registry
import file_utils  # Import the rest of file_utils
from mask_maker import run_mask_maker
from groundingdino.util.inference import load_model, load_image, predict, annotate  # <-- Added this line

# Download all necessary weights
file_utils.download_weights()

class Predictor(BasePredictor):
    def setup(self):
        """Set up the models and load them into memory."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setting up DINO model
        print("Setting up DINO model...")
        self.groundingdino_model = load_model(
            "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"{file_utils.WEIGHTS_INFO['DINO_WEIGHTS_DIR']}/groundingdino_swint_ogc.pth",
            device=self.device,
        )
        print("DINO model loaded successfully!")

        # Setting up SAM model
        print("Setting up SAM model...")
        try:
            sam_checkpoint = file_utils.WEIGHTS_INFO["SAM_WEIGHTS_LOCAL_FILE"]
            model_type = "vit_l"  # Adjust based on the model you're using
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print("SAM model loaded successfully!")
        except Exception as e:
            print(f"Failed to load SAM model: {e}")
            raise  # Re-raise the exception to prevent silent failures

    def predict(
            self,
            image: Path = Input(
                description="Image file path",
            ),
            mask_prompt: str = Input(
                description="Comma-separated mask terms (add a period at end for single object detection per term)",
                default="Person., shirt, pants, hat",
            ),
            threshold: float = Input(
                description="Threshold for object detection",
                default=0.2,
            )
    ) -> dict:
        """Run a single prediction and return mask data as a dictionary."""

        # Since 'image' is required, no default image handling is needed

        predict_id = str(uuid.uuid4())
        print(f"Running prediction ID: {predict_id}...")

        # Run mask maker and generate mask data
        mask_data = run_mask_maker(
            local_image_path=image,
            mask_prompt=mask_prompt,
            groundingdino_model=self.groundingdino_model,
            sam_predictor=self.sam_predictor,
            threshold=threshold
        )
        print(f"Generated mask data with {len(mask_data['terms'])} terms.")

        # Return the mask data directly
        return mask_data
