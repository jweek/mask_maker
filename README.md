## README

This model is a customized version of [grounding_sam](https://replicate.com/schananas/grounded_sam) with slight modifications to the input and output structure, and a custom run-length encoding (RLE) for mask compression.

[![Replicate](https://replicate.com/schananas/grounded_sam/badge)](https://replicate.com/schananas/grounded_sam)

### Overview

The model processes images to detect specific objects based on a **mask prompt** and returns masks in a custom, compact RLE format. This modified encoding optimizes storage and transmission, allowing efficient data handling.

---

### Input Fields

This model accepts the following input fields:

- **`image`**: The input image to process (file format).
- **`mask_prompt`**: A comma-separated list of object names to detect (e.g., `"vehicle., windows"`). A period after an object name (e.g., `vehicle.`) indicates that only the primary instance of that object should be detected.
- **`box_threshold`**: Confidence threshold for bounding boxes (float, default: `0.25`).
- **`text_threshold`**: Confidence threshold for text detections within the mask prompt (float, default: `0.25`).

---

### JSON Output

The model outputs JSON with the following structure:

```json
{
  "meta": {
    "request_id": "unique-request-id",
    "request_date": "YYYY-MM-DD HH:MM:SS",
    "processing_time": "total_processing_time_in_seconds",
    "dino_model": "GroundingDINO",
    "sam_model": "SAM",
    "dino_processing_time": "dino_processing_time_in_seconds",
    "sam_processing_time": "sam_processing_time_in_seconds",
    "mask_prompt": "vehicle., windows",
    "mask_encoding": "Custom RLE",
    "image_width": 4000,
    "image_height": 1800,
    "box_threshold": 0.25,
    "text_threshold": 0.25
  },
  "detections": [
    {
      "ratio": [cx, cy, w, h],           // Relative bounding box ratios
      "pixel": [x_min, y_min, x_max, y_max],  // Bounding box pixel coordinates
      "phrase": "detected_object_name",  
      "main_score": "composite_score",   // Scoring based on confidence, size, and proximity
      "confidence": "confidence_score"   // Initial confidence score from DINO
    },
  ],
  "terms": [
    {
      "term": "object_name",
      "single_flg": true,
      "mask_rle": {
        "size": [image_height, image_width],
        "counts": "RLE encoded mask counts"
      }
    }
  ]
}
```

### Custom RLE Encoding

Masks are encoded using a custom Run-Length Encoding (RLE) scheme, which provides compact storage for binary masks. The encoding works as follows:

1. **Counting Sequence**: The encoding starts in "black mode" (unmasked area) and counts pixels between switches (from black to white, or unmasked to masked areas).
2. **Encoding Limits**: `00` to `FE` represent counts from `0` to `254`, and `FFF` (hex) represents `255`, indicating the continuation of a single mode (i.e., 255 pixels of the current mode).
3. **Storage**: Counts are stored in a hexadecimal sequence, reducing storage space significantly. 

---

### Decoding the RLE Mask in Node.js

The following `decodeRle` method can be used in Node.js to decode the custom RLE format, returning a binary mask as a 1D Uint8Array:

```javascript
function decodeRle(rleMask, size) {
  const counts = rleMask.counts.match(/.{1,3}/g); // Split into 3-character hex groups
  const [height, width] = size;
  const numPixels = height * width;

  // Initialize the binary mask
  const binaryMask = new Uint8Array(numPixels);

  let idx = 0;
  let currentValue = 0; // Start with black (0)

  counts.forEach((hexCount) => {
    const count = parseInt(hexCount, 16);
    if (count === 0xFFF) {
      idx += 4095; // FFF represents a skip of 4095 pixels
    } else {
      for (let i = 0; i < count; i++) {
        if (idx >= numPixels) break;
        binaryMask[idx++] = currentValue;
      }
      currentValue = 1 - currentValue; // Toggle between 0 (black) and 1 (white)
    }
  });

  return binaryMask;
}
```

### Usage Example

This Node.js decoding method allows easy transformation of the custom RLE mask back into a usable binary format for downstream processing or visualization.

