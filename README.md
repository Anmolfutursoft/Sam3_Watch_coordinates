# Watch Dial Bounding Box Detection API

A FastAPI-based Computer Vision API that detects a watch using the **SAM3 segmentation model** served through **NVIDIA Triton Inference Server**, extracts the watch dial region, computes its bounding box, and returns either the coordinates or a downloadable annotated image.

# Features

- Detect watch and strap using SAM3
- Extract watch dial by removing the strap region
- Clean segmentation mask using image processing
- Compute dial bounding box
- Return bounding box coordinates as JSON
- Download annotated result image
- Triton Inference Server integration
- FastAPI REST API

# Requirements

- Python 3.12+
- NVIDIA Triton Inference Server
Python packages

```text
fastapi
uvicorn
numpy
opencv-python-headless
scipy
python-dotenv
python-multipart
tritonclient[grpc]
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Anmolfutursoft/Sam3_Watch_coordinates.git
cd Sam3_Watch_coordinates
```

Install dependencies and create the virtual environment using **uv**

```bash
uv sync
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Run the application

```bash
uv run uvicorn app:app --reload
```

---

# Environment Variables

Create a `.env` file in the project root.

```env
URL=<TRITON_SERVER_IP>
PORT=<TRITON_SERVER_PORT>
```
---

# Running the API

depending on your project structure.

API will be available at

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

---

## 1. Predict Bounding Box

**POST**

```
/bbox/predict
```

### Response

```json
{
    "success": true,
    "message": "Image loaded successfully. Size: 1920x1080",
    "bbox": [
        512,
        320,
        1048,
        856
    ]
}
```

---

## 2. Download Result Image

**POST**

```
/bbox/download
```

Returns

```
image/png
```

The downloaded image contains

- Watch segmentation
- Dial mask
- Bounding box
- Four corner coordinates
