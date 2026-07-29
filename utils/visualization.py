import cv2
import numpy as np

# This function creates a result image by overlaying the dial mask and bounding box on the original RGB image.
# It also marks the corners of the bounding box with circles and labels them (TL, TR, BL, BR).
#Step 1: Create a copy of the original RGB image to avoid modifying it directly.
#step 2: Create a watch image that only contains the pixels corresponding to the watch mask.
#Step 3: Draw the bounding box on the watch image using the coordinates provided in the bbox parameter.
#Step 4: Mark the corners of the bounding box with circles and label them
#Step 5: Create a mask image that only contains the pixels corresponding to the dial mask
#Step 6: Combine the watch image and the mask image side by side for visualization.     

def create_result_image(rgb_image, dial_mask, bbox, watch_mask):
    image = rgb_image.copy()
    
    watch_image = np.zeros_like(image)
    watch_image[watch_mask] = image[watch_mask]
    
    
    x1, y1, x2, y2 = bbox

    # Draw bbox
    cv2.rectangle(
        watch_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    corners = [
        ("TL", (x1, y1)),
        ("TR", (x2, y1)),
        ("BL", (x1, y2)),
        ("BR", (x2, y2)),
    ]

    for label, pt in corners:

        cv2.circle(watch_image, pt, 5, (255, 0, 0), -1)
        if label in ["TL", "BL"]:
            # Left corners -> text on right
            text_x = pt[0] + 10
        else:
            # Right corners -> text on left
            text_x = pt[0] - 425  

        if label in ["TL", "TR"]:
            # Top corners -> text below
            text_y = pt[1] + 45
        else:
            # Bottom corners -> text above
            text_y = pt[1] - 10


        cv2.putText(
            watch_image,
            f"{label}:{pt}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.75,
            (0,255,0),
            2
        )

    # Convert mask to RGB
    mask_image = np.zeros_like(image)
    mask_image[dial_mask] = rgb_image[dial_mask]

    # Combine images
    combined = np.hstack([watch_image, mask_image])

    return combined