import os
import sys
import subprocess
import torch
import json
from cog import BasePredictor, Input, Path
from typing import Iterator
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from grounded_sam import run_grounding_sam
import uuid
from hf_path_exports import cache_config_file, cache_file

# Environment setup for CUDA and Docker
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7'
os.environ['AM_I_DOCKER'] = 'true'
os.environ['BUILD_WITH_CUDA'] = 'true'

env_vars = os.environ.copy()
HOME = os.getcwd()

# Set up paths and install dependencies for GroundingDINO and Segment Anything
sys.path.insert(0, "weights")
sys.path.insert(0, "weights/GroundingDINO")
sys.path.insert(0, "weights/segment-anything")
os.chdir("weights/GroundingDINO")
subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir("weights/segment-anything")
subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir(HOME)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def load_model_hf(device='cpu'):
            args = SLConfig.fromfile(cache_config_file)
            args.device = device
            model = build_model(args)
            checkpoint = torch.load(cache_file, map_location=device)
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model

        # Load GroundingDINO model with weights
        self.groundingdino_model = load_model_hf(device)

        # Load SAM model with weights
        sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'  # Updated path for SAM checkpoint
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Image",
                default="https://st.mngbcn.com/rcs/pics/static/T5/fotos/outfit/S20/57034757_56-99999999_01.jpg",
            ),
            mask_prompt: str = Input(
                description="Mask Term List (limit mask to single object by adding a period at the end of the term)",
                default="vehicle., windows, wheels, headlights",
            ),
            box_threshold: float = Input(
                description="Box Threshold",
                default=0.3,
            ),
            text_threshold: float = Input(
                description="Text Threshold",
                default=0.25,
            )
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        predict_id = str(uuid.uuid4())
        print(f"Running prediction: {predict_id}...")

        # Generate mask data using GroundingDINO and SAM
        mask_data = run_grounding_sam(
            image,
            mask_prompt,
            self.groundingdino_model,
            self.sam_predictor,
            box_threshold,
            text_threshold
        )
        print("Generated mask data. size: ", len(mask_data))

        # Create output directory and save mask data
        output_dir = f"/tmp/{predict_id}"
        os.makedirs(output_dir, exist_ok=True)  # create directory if it doesn't exist
        mask_output_path = os.path.join(output_dir, "mask_data.json")

        with open(mask_output_path, 'w') as mask_file:
            json.dump(mask_data, mask_file)

        yield Path(mask_output_path)
