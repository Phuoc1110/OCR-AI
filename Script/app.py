#!/usr/bin/env python3
"""
Flask OCR server built from run1.py
Keeps the PaddleOCR model loaded in memory and exposes HTTP endpoints.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from paddleocr import PaddleOCR


BASE_DIR = Path(__file__).resolve().parent
DET_MODEL_DIR = BASE_DIR / "inference" / "det_mv3_db"
REC_MODEL_DIR = BASE_DIR / "inference" / "rec_japan_scratch_inference_18"
OUTPUT_DIR = BASE_DIR / "output"

import threading

app = Flask(__name__)
ocr = None
ocr_lock = threading.Lock()

def build_ocr():
    ocr_kwargs = {
        "det_model_dir": str(DET_MODEL_DIR),
        "rec_model_dir": str(REC_MODEL_DIR),
        "use_angle_cls": True,
        "lang": "japan",
        "show_log": False,
        "ir_optim": False,
        "enable_mkldnn": False,
    }

    return PaddleOCR(**ocr_kwargs)


def initialize_ocr():
    global ocr
    if ocr is None:
        print("[OCR Server] Initializing PaddleOCR model...", flush=True)
        print("[OCR Server] Using custom rec model.", flush=True)

        ocr = build_ocr()
        print("[OCR Server] PaddleOCR model initialized successfully!", flush=True)

    return ocr


def recognize_image_data(image_data):
    ocr_instance = initialize_ocr()

    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None or image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
        raise ValueError("Unable to decode image or image is too small")

    with ocr_lock:
        result = ocr_instance.ocr(image, cls=True)
        
    lines = result[0] if result else []

    recognized_items = []
    recognized_words = []

    if lines:
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

        image_data = image_file.read()
        payload = recognize_image_data(image_data)

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

        image_data = base64.b64decode(data["image"])
        payload = recognize_image_data(image_data)

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

    port = int(os.environ.get("PORT", 5001))
    print(f"[OCR Server] Starting OCR server on port {port}...", flush=True)
    # Disable threading because PaddleOCR C++ engine binds to the thread that first executed it
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)