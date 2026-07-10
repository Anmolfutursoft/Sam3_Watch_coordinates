from fastapi import UploadFile
import cv2
import io
import numpy as np
from scipy import ndimage

from utils.sam3 import sam3


def read_image(image: UploadFile) -> np.ndarray:
    
    ##Read an uploaded image and convert it to RGB Image.

    image_bytes = image.file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def detect_masks(rgb_image):
    return sam3.detect(
        image=rgb_image,
        prompt_list=["FULL WATCH", "STRAP"],
        threshold=0.25,
        mask_threshold=0.25,
    )


def get_watch_mask(decoded_masks_list, scores_list):
    watch_scores = np.array(scores_list[0])

    best_idx = np.argmax(watch_scores)

    watch_mask = decoded_masks_list[0][best_idx].astype(bool)

    return watch_mask
    
    
    
def get_strap_mask(decoded_masks_list):
    strap_mask = np.any(
        decoded_masks_list[1].astype(bool),
        axis=0
    )
    return strap_mask



def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    
    mask = mask.astype(bool)
    filled_mask = ndimage.binary_fill_holes(mask)

    return filled_mask


def keep_largest_components(mask: np.ndarray, k: int = 1) -> np.ndarray:
    labels, num = ndimage.label(mask)

    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = ndimage.sum(mask, labels, range(1, num + 1))

    order = np.argsort(sizes)[::-1][:k]

    result = np.zeros_like(mask, dtype=bool)

    for idx in order:
        result |= labels == (idx + 1)

    return result

def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask.astype(bool)

def get_dial_mask(watch_mask, strap_mask):
    return watch_mask & (~strap_mask)

def find_bbox(mask):
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]