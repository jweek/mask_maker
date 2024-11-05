import os
import torch
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
import requests

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE = "checkpoints"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

def download_weights(url: str, dest: str) -> None:
    if not os.path.exists(dest):
        print(f"Downloading weights from {url} to {dest}")
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(response.content)
        print("Weights downloaded successfully.")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the SAM model and prepare for inference."""
        global build_sam, SamPredictor
        from segment_anything import sam_model_registry, SamPredictor

        # Download weights if not already present
        weights_path = os.path.join(MODEL_CACHE, "sam_vit_b_01ec64.pth")
        download_weights(WEIGHTS_URL, weights_path)

        # Load the model
        self.sam_model = sam_model_registry["vit_b"](checkpoint=weights_path)
        self.sam_model.to(DEVICE)
        self.predictor = SamPredictor(self.sam_model)

    def predict(self, image: Path = Input(description="Input image")) -> Path:
        """Run a simple mask prediction using a center point in the image."""
        # Load and preprocess the image
        input_image = Image.open(image).convert("RGB")
        input_image_np = np.array(input_image)

        # Set the image in the predictor
        self.predictor.set_image(input_image_np)

        # Define a single point at the center of the image
        center_point = np.array([[input_image_np.shape[1] // 2, input_image_np.shape[0] // 2]])
        point_label = np.array([1], dtype=np.int32)  # Label for "foreground"

        # Generate the mask
        masks, _, _ = self.predictor.predict(point_coords=center_point, point_labels=point_label)

        # Convert the mask to uint8 format explicitly before saving
        mask_image = (masks[0].cpu().numpy() * 255).astype(np.uint8)  # Ensure uint8 dtype
        output_path = Path("output_mask.png")
        Image.fromarray(mask_image).save(output_path)

        return output_path

