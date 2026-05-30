#!/usr/bin/env python3
"""
Flask OCR server based on run1.py.
Uses PaddleOCR in memory and applies lightweight image normalization so
problematic uploads are less likely to hit native Paddle execution errors.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from paddleocr import PaddleOCR


BASE_DIR = Path(__file__).resolve().parent
DET_MODEL_DIR = BASE_DIR / "inference" / "det_mv3_db"
REC_MODEL_DIR = BASE_DIR / "inference" / "rec_japan_scratch_inference_5"
OUTPUT_DIR = BASE_DIR / "output"


app = Flask(__name__)
ocr = None


def use_custom_rec():
    value = os.environ.get("OCR_USE_CUSTOM_REC", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def build_ocr():
    ocr_kwargs = {
        "det_model_dir": str(DET_MODEL_DIR),
        "use_angle_cls": True,
        "lang": "japan",
        "show_log": False,
        "ir_optim": False,
        "enable_mkldnn": False,
    }

    if use_custom_rec():
        ocr_kwargs["rec_model_dir"] = str(REC_MODEL_DIR)

    return PaddleOCR(**ocr_kwargs)


def initialize_ocr():
    global ocr
    if ocr is None:
        print("[OCR Server] Initializing PaddleOCR model...", flush=True)
        if use_custom_rec():
            print("[OCR Server] Using custom rec model.", flush=True)
        else:
            print("[OCR Server] Using default rec model.", flush=True)

        ocr = build_ocr()
        print("[OCR Server] PaddleOCR model initialized successfully!", flush=True)

    return ocr


def normalize_image(image):
    if image is None:
        raise ValueError("Unable to decode image")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image = np.ascontiguousarray(image)

    min_side = min(image.shape[0], image.shape[1])
    if min_side < 32:
        scale = 32.0 / float(min_side)
        new_width = max(32, int(image.shape[1] * scale))
        new_height = max(32, int(image.shape[0] * scale))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return image


def run_ocr_from_bytes(image_data):
    ocr_instance = initialize_ocr()

    nparr = np.frombuffer(image_data, np.uint8)
    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    image = normalize_image(decoded_image)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image)

        result = ocr_instance.ocr(temp_path, cls=True)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    lines = result[0] if result else []

    recognized_items = []
    recognized_words = []

    for line in lines:
        if not line or len(line) < 2:
            continue

        box = np.array(line[0]).astype(np.int32)
        text = str(line[1][0]).strip() if line[1] else ""
        confidence = float(line[1][1]) if line[1] and len(line[1]) > 1 else 0.0

        if text:
            recognized_items.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "box": box.tolist(),
                }
            )
            recognized_words.append(text)
            cv2.polylines(image, [box], isClosed=True, color=(0, 0, 255), thickness=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "result_with_boxes.jpg"), image)

    unique_words = []
    seen = set()
    for word in recognized_words:
        normalized = word.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_words.append(normalized)

    return {
        "words": unique_words,
        "text": " ".join(unique_words).strip(),
        "items": recognized_items,
    }


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "OCR Server is running"})


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    try:
        if "image" not in request.files:
            return (
                jsonify(
                    {
                        "errCode": 1,
                        "errMessage": "Missing image file",
                        "words": [],
                        "text": "",
                        "items": [],
                    }
                ),
                400,
            )

        image_file = request.files["image"]
        if not image_file or image_file.filename == "":
            return (
                jsonify(
                    {
                        "errCode": 1,
                        "errMessage": "Empty image file",
                        "words": [],
                        "text": "",
                        "items": [],
                    }
                ),
                400,
            )

        payload = run_ocr_from_bytes(image_file.read())

        return jsonify(
            {
                "errCode": 0,
                "errMessage": "OK",
                "words": payload.get("words", []),
                "text": payload.get("text", ""),
                "items": payload.get("items", []),
            }
        )

    except Exception as e:
        print(f"[OCR Server] Error: {str(e)}", flush=True)
        return (
            jsonify(
                {
                    "errCode": -1,
                    "errMessage": f"Internal server error: {str(e)}",
                    "words": [],
                    "text": "",
                    "items": [],
                }
            ),
            500,
        )


@app.route("/ocr-base64", methods=["POST"])
def ocr_base64_endpoint():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return (
                jsonify(
                    {
                        "errCode": 1,
                        "errMessage": "Missing image data",
                        "words": [],
                        "text": "",
                        "items": [],
                    }
                ),
                400,
            )

        import base64

        payload = run_ocr_from_bytes(base64.b64decode(data["image"]))

        return jsonify(
            {
                "errCode": 0,
                "errMessage": "OK",
                "words": payload.get("words", []),
                "text": payload.get("text", ""),
                "items": payload.get("items", []),
            }
        )

    except Exception as e:
        print(f"[OCR Server] Error: {str(e)}", flush=True)
        return (
            jsonify(
                {
                    "errCode": -1,
                    "errMessage": f"Internal server error: {str(e)}",
                    "words": [],
                    "text": "",
                    "items": [],
                }
            ),
            500,
        )


if __name__ == "__main__":
    initialize_ocr()

    port = int(os.environ.get("OCR_SERVER_PORT", 5001))
    print(f"[OCR Server] Starting OCR server on port {port}...", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False)