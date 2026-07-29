import io

import cv2
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from utils.schema import SegmentationResponse
from utils.visualization import create_result_image
from utils.components import (
    get_dial_mask,
    oppening_mask,
    read_image,
    get_watch_mask,
    get_strap_mask,
    find_bbox,
    keep_largest_components,
    save_mask
)
import numpy as np

router = APIRouter(
    prefix="/bbox",
    tags=["Bounding Box"]
)
from utils.infer_util import detect_masks

#this function is executed when a POST request is sent to /bbox/predict. 
# It takes an image as input, processes it to detect masks, and 
# returns the bounding box and other relevant information in the response.

@router.post("/predict", response_model=SegmentationResponse)
async def predict_bbox(image: UploadFile = File(...)):
    try:
        rgb_image = read_image(image)
        
        # Detect masks baiscly we are calling the detect_masks function from 
        # utils/infer_util.py to get the decoded masks, scores, and bounding boxes for the input image.

        decoded_masks_list, scores_list, boxes_list = detect_masks(rgb_image)
        
        # Get the watch mask and strap mask from the decoded masks and scores.

        watch_mask = get_watch_mask(decoded_masks_list, scores_list)
        strap_mask = get_strap_mask(decoded_masks_list)
        
        # Get the dial mask by combining the watch mask and strap mask, and then process it to find the bounding box.
        dial_mask = get_dial_mask(watch_mask, strap_mask) 
        
        # Process the dial mask to keep only the largest components and apply morphological opening to refine the mask.
        dial_mask = keep_largest_components(dial_mask) 
        dial_mask = oppening_mask(dial_mask)
        
        # Find the bounding box of the dial mask using the find_bbox function.
        bbox = find_bbox(dial_mask)
        
        
        # Return a SegmentationResponse object containing the success status, message, bounding box, and lists of boxes and scores.
        return SegmentationResponse(
            success=True,
            message=f"Image loaded successfully. Size: {rgb_image.shape[1]}x{rgb_image.shape[0]}",
            bbox=bbox,
            boxes_list=boxes_list,
            scores_list=scores_list
        )
        
    # Handle any exceptions that occur during the processing and return an error response.    
    except Exception as e:
        return SegmentationResponse(
            success=False,
            message=str(e),
            bbox=None,
            boxes_list=None
        )
        
 
        
# This function is executed when a POST request is sent to /bbox/download.
# It takes an image as input, processes it to detect masks, and returns the result image        
@router.post("/download")
async def download_result(image: UploadFile = File(...)):

    rgb_image = read_image(image)

    decoded_masks_list, scores_list, boxes_list = detect_masks(rgb_image)

    watch_mask = get_watch_mask(decoded_masks_list, scores_list)
    strap_mask = get_strap_mask(decoded_masks_list)

    dial_mask = get_dial_mask(watch_mask, strap_mask)
    dial_mask = keep_largest_components(dial_mask)
    dial_mask = oppening_mask(dial_mask)

    bbox = find_bbox(dial_mask)
    
    # Create the result image by overlaying the dial mask and bounding box on the original image.

    result = create_result_image(rgb_image, dial_mask, bbox, watch_mask)

    _, buffer = cv2.imencode(".png",cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=watch_result.png"
        }
    )