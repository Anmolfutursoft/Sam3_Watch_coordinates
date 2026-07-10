import os
import numpy as np
import base64
import zlib
from tritonclient import grpc as grpcclient
import json


url = os.getenv("TRITON_SERVER_URL", "localhost:8000")
print("Triton URL:", url)

class SAM3:
    def __init__(self):
        self.client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
        )

    def detect(self, image, prompt_list=[None], threshold=0.25, mask_threshold=0.25):
        # img_array = cv2.imread(image_path)
        # img_array = np.array([img_array[:, :, ::-1]], dtype=np.uint8)
        # Prepare the Triton input
        images = []
        for prompt in prompt_list:
            images.append(image)
        images = np.array(
            images,
            dtype=np.uint8,
        )
        sam3_data = {
            "prompt_list": prompt_list,
            "bboxes_list": [None],
            "labels_list": [None],
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }

        # sam3_data = process_sam3_data(sam3_data)
        # print("SAM3 data")
        # print(sam3_data)
        input_data = json.dumps(sam3_data)
        input_data = np.array([input_data], dtype=np.object_)

        input_images = grpcclient.InferInput("images", images.shape, "UINT8")
        input_images.set_data_from_numpy(images)

        inputs = grpcclient.InferInput("sam3_input", [1], "BYTES")
        inputs.set_data_from_numpy(input_data)

        # Specify output

        output_mask = grpcclient.InferRequestedOutput("sam3_result")

        # Call the ensemble model

        # try:
        response = self.client.infer(
            model_name="sam3",
            inputs=[inputs, input_images],
            outputs=[output_mask],
            compression_algorithm="gzip",
        )

        # Extract NumPy arrays

        sam3_result_bytes = response.as_numpy("sam3_result")[0]

        # Step 1: Decode bytes to string
        # Step 2: Convert JSON string to Python dict
        json_str = sam3_result_bytes.decode("utf-8")

        # Step 2: Convert JSON string to Python dict
        sam3_result = json.loads(json_str)
        # print("--- SAM3 Result ---")
        # print(sam3_result)
        # idx = sam3_result[0]['scores'].index(max(sam3_result[0]['scores']))
        masks_dict_list = []
        scores_list = []
        boxes_list = []
        for idx, result in enumerate(sam3_result):
            masks_dict = result["masks"]
            masks_dict_list.append(masks_dict)
            scores = result["scores"]
            scores_list.append(scores)
            boxes = result.get("boxes", result.get("bboxes", []))
            boxes_list.append(boxes)
            print(f"Image {idx}: {len(boxes)} bbox(es)")
            for b_idx, box in enumerate(boxes):
                print(f"  box {b_idx}: {box}")
        decoded_masks_list = []
        for masks_dict in masks_dict_list:
            decoded_masks = self.decode_mask(masks_dict)
            decoded_masks_list.append(decoded_masks)

        # mask = decoded_masks[idx]
        # union_mask = np.any(decoded_masks, axis=0)

        return decoded_masks_list, scores_list, boxes_list

    @staticmethod
    def decode_mask(mask_dict):
        shape = tuple(mask_dict["shape"])
        compressed = base64.b64decode(mask_dict["b64"])
        raw = zlib.decompress(compressed)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape)

        return arr.astype(np.uint8)

    @staticmethod
    def mask_to_bbox(mask):
        """Compute the tight (x1, y1, x2, y2) bounding box of a 2D mask.

        Returns None if the mask is empty. The box is derived from the mask
        itself, so it always aligns with what gr.AnnotatedImage renders.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        # +1 on the far edges so the box fully encloses the last pixel
        return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]
    
sam3 = SAM3()