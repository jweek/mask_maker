import os
import sys
import subprocess
import torch

#install GroundingDINO and segment_anything
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ['AM_I_DOCKER'] = 'true'
os.environ['BUILD_WITH_CUDA'] = 'true'
os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

env_vars = os.environ.copy()
HOME = os.getcwd()
sys.path.insert(0, "weights")
sys.path.insert(0, "weights/GroundingDINO")
sys.path.insert(0, "weights/segment-anything")
os.chdir("/src/weights/GroundingDINO")
subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir("/src/weights/segment-anything")
subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir(HOME)

from cog import BasePredictor, Input, Path, BaseModel
from typing import Iterator
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from grounded_sam import run_grounding_sam
import uuid
from hf_path_exports import cache_config_file, cache_file

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")

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

        self.groundingdino_model = load_model_hf(device)
        sam_checkpoint = '/src/weights/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Image",
                default="https://st.mngbcn.com/rcs/pics/static/T5/fotos/outfit/S20/57034757_56-99999999_01.jpg",
            ),
            mask_prompt: str = Input(
                description="Positive mask prompt",
                default="clothes,shoes",
            ),
            negative_mask_prompt: str = Input(
                description="Negative mask prompt",
                default="pants",
            ),
            adjustment_factor: int = Input(
                description="Mask Adjustment Factor (-ve for erosion, +ve for dilation)",
                default=0,
            ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        predict_id = str(uuid.uuid4())

        print(f"Running prediction: {predict_id}...")

        annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = run_grounding_sam(image,
                                                                                                    mask_prompt,
                                                                                                    negative_mask_prompt,
                                                                                                    self.groundingdino_model,
                                                                                                    self.sam_predictor,
                                                                                                    adjustment_factor)
        print("Done!")

        variable_dict = {
            'annotated_picture_mask': annotated_picture_mask,
            'neg_annotated_picture_mask': neg_annotated_picture_mask,
            'mask': mask,
            'inverted_mask': inverted_mask
        }

        output_dir = "/tmp/" + predict_id
        os.makedirs(output_dir, exist_ok=True)  # create directory if it doesn't exist

        for var_name, img in variable_dict.items():
            random_filename = output_dir + "/" + var_name + ".jpg"
            rgb_img = img.convert('RGB')  # Converting image to RGB
            rgb_img.save(random_filename)
            yield Path(random_filename)