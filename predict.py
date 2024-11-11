# predict.py

import os

from file_utils import WEIGHTS_INFO  # Import WEIGHTS_INFO without importing transformers

# Set environment variables before any other imports
os.environ["HF_HOME"] = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]
os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]
os.environ["TRANSFORMERS_CACHE"] = WEIGHTS_INFO["HUGGINGFACE_CACHE_DIR"]

# Now import the rest
from cog import BasePredictor, BaseModel, Input, Path
from typing import Optional, List
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from cv2 import imwrite as cv2_imwrite
import file_utils  # Import the rest of file_utils
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Download all necessary weights
file_utils.download_weights()

class ModelOutput(BaseModel):
    detections: List
    result_image: Optional[Path]
    sam_masks_generated: bool  # Simple flag for SAM
    error_message: Optional[str]  # Capture error message

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Set up the models and load them into memory."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setting up DINO model
        print("Setting up DINO model...")
        self.model = load_model(
            "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"{file_utils.WEIGHTS_INFO['WEIGHTS_CACHE_DIR']}/groundingdino_swint_ogc.pth",
            device=self.device,
        )
        print("DINO model loaded successfully!")

        # Setting up SAM model
        print("Setting up SAM model...")
        self.sam_model = sam_model_registry["vit_l"](checkpoint=file_utils.WEIGHTS_INFO["SAM_WEIGHTS_LOCAL_FILE"])
        self.sam_mask_generator = SamAutomaticMaskGenerator(self.sam_model)
        self.sam_model.to(self.device)
        print("SAM model loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Input image to query", default=None),
        query: str = Input(
            description="Comma-separated names of the objects to be detected in the image",
            default="car",  # Provide a default query in case it's not provided
        ),
        box_threshold: float = Input(
            description="Confidence level for object detection",
            ge=0,
            le=1,
            default=0.25,
        ),
        text_threshold: float = Input(
            description="Confidence level for object detection",
            ge=0,
            le=1,
            default=0.25,
        ),
        show_visualisation: bool = Input(
            description="Draw and visualize bounding boxes on the image", default=True
        ),
    ) -> ModelOutput:
        error_message = None
        try:
            # Load image
            image_source, image = load_image(image)

            # DINO predictions
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=query,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

            # Convert boxes from center, width, height to top left, bottom right
            height, width, _ = image_source.shape
            boxes_original_size = boxes * torch.Tensor([width, height, width, height])
            xyxy = (
                box_convert(boxes=boxes_original_size, in_fmt="cxcywh", out_fmt="xyxy")
                .numpy()
                .astype(int)
            )

            # Prepare DINO output
            detections = []
            for box, score, label in zip(xyxy, logits, phrases):
                data = {
                    "label": label,
                    "confidence": score.item(),  # torch tensor to float
                    "bbox": box,
                }
                detections.append(data)

            # Run SAM on the image to generate masks
            sam_masks = self.sam_mask_generator.generate(image_source)
            sam_masks_generated = len(sam_masks) > 0

            # Optional visualization of DINO output
            result_image_path = None
            if show_visualisation:
                result_image_path = "/tmp/result.png"
                result_image = annotate(
                    image_source=image_source,
                    boxes=boxes,
                    logits=logits,
                    phrases=phrases,
                )
                cv2_imwrite(result_image_path, result_image)

        except Exception as e:
            # Capture the error message and set the SAM masks flag to False
            error_message = str(e)
            sam_masks_generated = False
            result_image_path = None
            detections = []

        # Return output with error message if any exception occurred
        return ModelOutput(
            detections=detections,
            result_image=None,
            sam_masks_generated=sam_masks_generated,
            error_message=error_message,
        )
