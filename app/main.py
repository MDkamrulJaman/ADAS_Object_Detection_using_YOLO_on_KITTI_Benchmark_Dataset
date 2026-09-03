import io
import logging
import os
import json

import cv2
import numpy as np
import gradio as gr
import requests
from PIL import Image

from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from pathlib import Path
from dotenv import load_dotenv



# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("gradio-wrapper")


# ============================================================
# Configuration
# ============================================================

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)



MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

PORT = int(os.getenv("PORT", "8000"))

MODEL_API_TIMEOUT = int(
    os.getenv("MODEL_API_TIMEOUT", "120")
)


# ============================================================
# Validate required configuration
# ============================================================

if not MODEL_API_URL:
    raise RuntimeError(
        "MODEL_API_URL environment variable is not set."
    )

if not MODEL_API_KEY:
    raise RuntimeError(
        "MODEL_API_KEY environment variable is not set."
    )


logger.info("Model API configured: %s", MODEL_API_URL)


# ============================================================
# Class Names Mapping
# ============================================================

CLASS_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "Person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}


# ============================================================
# Model API Inference
# ============================================================

def predict_image(
    img,
    conf_threshold,
    iou_threshold,
):
    """
    Send the uploaded image to the existing live
    YOLO inference API.

    The YOLO model itself is NOT loaded here.
    """

    if img is None:
        raise gr.Error(
            "Please upload an image."
        )

    try:
        image_buffer = io.BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(image_buffer, format="JPEG", quality=95)
        image_buffer.seek(0)

        files = {"image": ("image.jpg", image_buffer, "image/jpeg")}
        data = {
            "conf_threshold": str(conf_threshold),
            "iou_threshold": str(iou_threshold),
        }
        headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}

        response = requests.post(
            MODEL_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=MODEL_API_TIMEOUT,
        )

        if not response.ok:
            logger.error("API returned HTTP %s", response.status_code)
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text
            raise gr.Error(f"API error ({response.status_code}): {error_detail}")

        result_json = response.json()
        if "images" not in result_json or not result_json["images"]:
            raise gr.Error("Invalid API response")

        detections = result_json["images"][0].get("results", [])

        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # BGR colors for OpenCV
        colors_bgr = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
        }
        color_names = list(colors_bgr.keys())
        
        for idx, det in enumerate(detections):
            if "box" in det:
                box = det["box"]
                x1 = int(box.get("x1", 0))
                y1 = int(box.get("y1", 0))
                x2 = int(box.get("x2", 0))
                y2 = int(box.get("y2", 0))
                
                if all(c for c in [x1, y1, x2, y2]):
                    color_name = color_names[idx % len(color_names)]
                    color = colors_bgr[color_name]
                    
                    # Draw bounding box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Get confidence
                    conf = det.get("confidence") or det.get("conf") or det.get("score") or 0
                    
                    # Get class - try multiple field names
                    class_id = det.get("class_id") or det.get("class") or 0
                    class_name = det.get("name") or CLASS_NAMES.get(int(class_id), "Unknown")
                    
                    logger.info(f"Detection {idx}: {det}")
                    label = f"{class_name} {conf:.2f}"
                    
                    # Put text with background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Position text above box
                    text_x = x1
                    text_y = max(text_size[1] + 5, y1 - 5)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(
                        cv_image,
                        (text_x - 2, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 2, text_y + 3),
                        color,
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        cv_image,
                        label,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )

        # Convert back to PIL
        result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        logger.info(f"Detected {len(detections)} objects")
        return result_image

    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        raise gr.Error("Request timed out. Please try again.")

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API")
        raise gr.Error("Could not connect to API.")

    except requests.exceptions.RequestException as exc:
        logger.exception("API request failed")
        raise gr.Error(f"Request failed: {exc}")

    except gr.Error:
        raise

    except Exception as exc:
        logger.exception("Unexpected error")
        raise gr.Error(f"Error: {exc}")


# ============================================================
# Gradio Interface
# ============================================================

demo = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.70, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Multi-Class Object Detection Model for Self-Driving Vehicle Safety",
    description="Upload images for inference via Ultralytics API.",
    api_name="predict",

)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="YOLO26 Wrapper", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "gradio-wrapper"}

app = mount_gradio_app(app, demo, path="/")

