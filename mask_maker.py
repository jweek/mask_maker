from PIL import Image
import numpy as np
import torch
import uuid
from datetime import datetime
import time

from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, predict

class MaskTerm:
    def __init__(self, term, single_flg):
        self.term = term
        self.single_flg = single_flg
        self.mask_data = None  # To store the custom RLE-encoded mask data

def custom_rle_encode(flat_mask):
    """
    Custom RLE encoding for a binary mask using a three-character hexadecimal scheme.
    Each count represents the run length of a color, starting with black (0).
    'FFF' is used to indicate skipping 4095 pixels without a color switch.
    """
    counts = []
    current_value = 0  # Start with black
    count = 0

    for pixel in flat_mask:
        if pixel == current_value:
            if count == 4095:  # Maximum value for three hexadecimal digits
                counts.append("FFF")  # Use FFF to represent max skip
                count = 1  # Reset count and keep the same color
            else:
                count += 1
        else:
            counts.append(f"{count:03X}")  # Convert to three-character hexadecimal
            count = 1
            current_value = pixel

    # Append the final count
    counts.append(f"{count:03X}")

    return ''.join(counts)

def score_detection(detection, image_width, image_height):
    """
    Scores a detection based on confidence, size, and proximity to center with adjustable weights.
    Args:
        detection (dict): Detection dictionary containing "confidence" and "pixel" (bounding box in pixels).
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
    Returns:
        float: Composite score of the detection.
    """
    w_conf = 0.5
    w_size = 0.5
    w_proximity = 0.2

    # Extract detection properties
    confidence = detection["confidence"]
    x_min, y_min, x_max, y_max = detection["pixel"]

    # Calculate size (area) of the bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min
    box_area = box_width * box_height

    # Calculate distance of the box center to the image center
    box_center_x = x_min + box_width / 2
    box_center_y = y_min + box_height / 2
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    distance_to_center = ((box_center_x - image_center_x) ** 2 + (box_center_y - image_center_y) ** 2) ** 0.5

    # Normalize size and distance metrics
    normalized_size = box_area / (image_width * image_height)
    normalized_distance = 1 - (distance_to_center / ((image_width ** 2 + image_height ** 2) ** 0.5))

    # Adjust confidence score with thresholds
    if confidence >= 0.4:
        adj_confidence = 0.4
    elif confidence < 0.3:
        adj_confidence = confidence - 0.1
    else:
        adj_confidence = confidence

    # Adjust size score
    if normalized_size >= 0.5:
        adj_size = 0.5
    elif normalized_size < 0.3:
        adj_size = normalized_size - 0.1
    else:
        adj_size = normalized_size

    # Composite score with adjustable weights
    score = (adj_confidence * w_conf) + (adj_size * w_size) + (normalized_distance * w_proximity)
    return score

def run_mask_maker(local_image_path, mask_prompt, groundingdino_model, sam_predictor, box_threshold, text_threshold):
    start_time = time.time()

    # Parse the mask_prompt into a list of MaskTerm objects
    terms = [MaskTerm(term.strip().rstrip('.'), term.endswith('.')) for term in mask_prompt.split(',')]

    # Load image
    image_source, image = load_image(local_image_path)
    height, width, _ = image_source.shape 

    # Process all terms in one call to DINO
    mark_time = time.time()
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=mask_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    dino_processing_time = time.time() - mark_time

    # Prepare detections data using box_ops for pixel conversion
    detections = []
    term_to_detections = {term.term: [] for term in terms}  # Map term to its detections
    for i, box in enumerate(boxes):
        # Convert box from center-based coordinates to corner-based coordinates
        box_xyxy = box_ops.box_cxcywh_to_xyxy(box)

        # Scale box to pixel coordinates
        pixel_box = box_xyxy * torch.Tensor([width, height, width, height])

        # Ensure pixel values are within image bounds
        x_min, y_min, x_max, y_max = pixel_box.int().tolist()
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(0, min(x_max, width - 1))
        y_max = max(0, min(y_max, height - 1))

        # Calculate main_score and store it in detection_data
        detection_data = {
            "ratio": box.tolist(),          # Original ratio coordinates [cx, cy, width, height]
            "pixel": [x_min, y_min, x_max, y_max],  # Converted pixel coordinates [x_min, y_min, x_max, y_max]
            "phrase": phrases[i],           # Detected phrase
            "confidence": logits[i].item(),  # Confidence score as a float
            "main_score": score_detection(
                {"confidence": logits[i].item(), "pixel": [x_min, y_min, x_max, y_max]},
                width,
                height
            )  # Main score as a float
        }
        detections.append(detection_data)

        # Map detections to terms
        term_to_detections[phrases[i]].append(detection_data)

    # SAM Mask Processing per Term
    mark_time = time.time()
    for term in terms:
        detections_for_term = term_to_detections.get(term.term, [])
        if not detections_for_term:
            continue  # No detections for this term

        # Handle single object selection
        if term.single_flg:
            main_detection = max(
                detections_for_term,
                key=lambda det: det["main_score"]
            )
            boxes_for_sam = [main_detection['pixel']]
        else:
            boxes_for_sam = [det['pixel'] for det in detections_for_term]

        # Prepare boxes for SAM
        sam_boxes = torch.tensor(boxes_for_sam, dtype=torch.float32).to(sam_predictor.device)

        # Generate masks using SAM
        sam_predictor.set_image(image_source)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(sam_boxes, image_source.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Merge masks if multiple
        merged_mask = np.logical_or.reduce(masks.squeeze(1).cpu().numpy()) if masks.shape[0] > 1 else masks[0, 0].cpu().numpy()

        # Flatten the mask to a 1D array and apply custom RLE encoding
        flat_mask = merged_mask.flatten().astype(np.uint8)
        rle_counts = custom_rle_encode(flat_mask)

        # Store the custom RLE counts in the MaskTerm object
        term.mask_data = {
            'size': [height, width],
            'counts': rle_counts
        }

    sam_processing_time = time.time() - mark_time
    processing_time = time.time() - start_time

    # Prepare data for output
    term_data_arr = [
        {
            "term": term.term,
            "single_flg": term.single_flg,
            "mask_rle": term.mask_data
        }
        for term in terms
    ]

    meta_data = {
        "request_id": str(uuid.uuid4()),
        "request_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time": f"{processing_time:.4f}",
        "dino_model": "GroundingDINO",
        "sam_model": "SAM",
        "dino_processing_time": f"{dino_processing_time:.4f}",
        "sam_processing_time": f"{sam_processing_time:.4f}",
        "mask_prompt": mask_prompt,
        "mask_encoding": "Custom RLE",
        "image_width": width,
        "image_height": height,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold
    }

    # Updated mask_data with the new JSON output
    mask_data = {
        "meta": meta_data,
        "detections": detections,
        "terms": term_data_arr
    }

    return mask_data
