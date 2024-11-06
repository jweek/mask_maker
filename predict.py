from cog import BasePredictor, BaseModel, Input, Path
import os
from typing import Optional, List
import torch
from cv2 import imwrite as cv2_imwrite
import file_utils
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Define weight paths and directories
WEIGHTS_CACHE_DIR = "/src/weights"
SAM2_WEIGHTS_DIR = "/src/sam2_weights"
HUGGINGFACE_CACHE_DIR = "/src/hf-cache/"
os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = HUGGINGFACE_CACHE_DIR

# Download weights
file_utils.download_grounding_dino_weights(grounding_dino_weights_dir=WEIGHTS_CACHE_DIR, hf_cache_dir=HUGGINGFACE_CACHE_DIR)
file_utils.download_sam2_tiny_weights(sam2_weights_dir=SAM2_WEIGHTS_DIR)

class ModelOutput(BaseModel):
    detections: List
    result_image: Optional[Path]
    sam2_status: str  # Added to include SAM-2 status

class Predictor(BasePredictor):
    def setup(self) -> None:
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load GroundingDINO model
        self.dino_model = load_model(
            "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"{WEIGHTS_CACHE_DIR}/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        # Attempt to initialize SAM-2 model
        try:
            sam2_checkpoint = os.path.join(SAM2_WEIGHTS_DIR, "sam2.1_hiera_tiny.pt")
            model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
            self.sam2_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint).to(self.device))
            self.sam2_initialized = True
            print("SAM-2 initialized successfully.")
        except Exception as e:
            self.sam2_initialized = False
            print("Failed to initialize SAM-2:", e)

    def predict(
        self,
        image: Path = Input(description="Input image to query", default=None),
        query: str = Input(description="Comma-separated names of objects to detect", default=None),
        box_threshold: float = Input(description="Confidence for object detection", ge=0, le=1, default=0.25),
        text_threshold: float = Input(description="Confidence for text", ge=0, le=1, default=0.25),
        show_visualisation: bool = Input(description="Visualize bounding boxes on the image", default=True),
    ) -> ModelOutput:
        # DINO prediction logic
        image_source, image = load_image(image)

        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # Convert boxes for output
        height, width, _ = image_source.shape
        boxes_original_size = boxes * torch.Tensor([width, height, width, height])
        xyxy = box_convert(boxes=boxes_original_size, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)

        # Prepare output detections
        detections = []
        for box, score, label in zip(xyxy, logits, phrases):
            data = {
                "label": label,
                "confidence": score.item(),
                "bbox": box,
            }
            detections.append(data)

        # Optional visualization
        result_image_path = None
        if show_visualisation:
            result_image_path = "/tmp/result.png"
            result_image = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2_imwrite(result_image_path, result_image)

        # Confirm SAM-2 installation
        sam2_status = "SAM-2 is initialized and ready." if self.sam2_initialized else "SAM-2 is not initialized."

        return ModelOutput(
            detections=detections,
            result_image=Path(result_image_path) if show_visualisation else None,
            sam2_status=sam2_status,
        )
