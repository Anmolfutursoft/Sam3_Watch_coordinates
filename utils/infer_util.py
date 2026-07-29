from utils.sam3 import SAM3
from utils.static_values import (
    PROMPT,
    THRESHOLD,
    MASK_THRESHOLD,
)
# Here we are creating an instance of the SAM3 class, which is responsible for handling the segmentation tasks. 
# This instance will be used to call the detect method to perform mask detection on the input images.
sam3 = SAM3()

# This function takes an image as input and uses the SAM3 instance to detect masks, scores, and bounding boxes.


def detect_masks(image):
    # Get raw response from Triton
    # The detect method of the SAM3 instance is called with the input image and some predefined parameters (PROMPT, THRESHOLD, MASK_THRESHOLD).
    # This method returns a list of results, where each result contains information about detected masks, scores, and bounding boxes for the input image.
    # scores = [0.95, 0.91] for each detected object, list of floats.
    # boxes = [[x1, y1, x2, y2], ...] for each detected object, list of 4 floats.
    # masks = {"shape": [num_masks, height, width], "b64": "base64_encoded_mask_data"}
    

    sam3_result = sam3.detect(
        image=image,
        prompt_list=PROMPT,
        threshold=THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
    )
    decoded_masks_list = []
    scores_list = []
    boxes_list = []
    
    for result in sam3_result:
         # Decode mask
         # 
        decoded_masks = sam3.decode_mask(result["masks"])
        decoded_masks_list.append(decoded_masks)
        
        # Extract scores
        scores_list.append(result["scores"])

        # Extract boxes
        boxes = result.get("boxes", result.get("bboxes", []))
        boxes_list.append(boxes)
        
    

    return decoded_masks_list, scores_list, boxes_list
    
    
##'[{"scores":[0.98],"boxes":[[10,20,100,150]],"masks":{"shape":[3,640,480],"b64":"eJz..."}}]'
'''[
    {
        "scores": [0.95, 0.91],
        "boxes": [
            [100, 50, 200, 300],
            [120, 80, 220, 310]
        ],
        "masks": {
            "shape": [2, 640, 480],
            "b64": "eJz..."
        }
    },
    {
        "scores": [0.98],
        "boxes": [
            [110, 60, 210, 320]
        ],
        "masks": {
            "shape": [1, 640, 480],
            "b64": "eJz..."
        }
    }
]'''