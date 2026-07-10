from fastapi import APIRouter, UploadFile, File
from utils.schema import SegmentationResponse
from utils.components import get_dial_mask, read_image , get_watch_mask, get_strap_mask, fill_mask_holes, smooth_mask, find_bbox, keep_largest_components, detect_masks

router = APIRouter(
    prefix="/bbox",
    tags=["Bounding Box"]
)

@router.post("/predict", response_model=SegmentationResponse)
async def predict_bbox(image: UploadFile = File(...)):
    try:
        rgb_image = read_image(image)

        decoded_masks_list, scores_list, boxes_list = detect_masks(rgb_image)

        watch_mask = get_watch_mask(decoded_masks_list, scores_list)
        strap_mask = get_strap_mask(decoded_masks_list)
        watch_mask = keep_largest_components(watch_mask)
        watch_mask = fill_mask_holes(watch_mask)
        watch_mask = smooth_mask(watch_mask)
        strap_mask = fill_mask_holes(strap_mask)
        strap_mask = smooth_mask(strap_mask)
        strap_mask = keep_largest_components(strap_mask, k=2)

        dial_mask = get_dial_mask(watch_mask, strap_mask)

        bbox = find_bbox(dial_mask)

        return SegmentationResponse(
            success=True,
            message=f"Image loaded successfully. Size: {rgb_image.shape[1]}x{rgb_image.shape[0]}",
            bbox=bbox
        )
    except Exception as e:
        return SegmentationResponse(
            success=False,
            message=str(e),
            bbox=None,
        )
        
        