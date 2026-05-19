#!/usr/bin/env python3
"""
OCR Server - Flask-based service to keep PaddleOCR model in memory
This prevents model from being reloaded on every request
"""

import json
import os
import sys
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
DET_MODEL_DIR = BASE_DIR / "inference" / "det_mv3_db"
REC_MODEL_DIR = BASE_DIR / "inference" / "rec_japan_scratch_inference_5"
OUTPUT_DIR = BASE_DIR / "output"

# Initialize Flask app
app = Flask(__name__)

# Global OCR instance (loaded once on server startup)
ocr = None


def use_custom_rec():
    value = os.environ.get("OCR_USE_CUSTOM_REC", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def initialize_ocr():
    """Initialize PaddleOCR model once"""
    global ocr
    if ocr is None:
        print("[OCR Server] Initializing PaddleOCR model...", flush=True)
        ocr_kwargs = {
            "det_model_dir": str(DET_MODEL_DIR),
            "use_angle_cls": True,
            "lang": "japan",
            "show_log": False,
        }

        if use_custom_rec():
            ocr_kwargs["rec_model_dir"] = str(REC_MODEL_DIR)
            print("[OCR Server] Using custom rec model.", flush=True)
        else:
            print("[OCR Server] Using default rec model.", flush=True)

        ocr = PaddleOCR(**ocr_kwargs)
        print("[OCR Server] PaddleOCR model initialized successfully!", flush=True)
    return ocr


def recognize_image_data(image_data):
    """
    Process image data and recognize text
    
    Args:
        image_data: Binary image data
    
    Returns:
        Dictionary with recognized words, text, and bounding boxes
    """
    ocr_instance = initialize_ocr()
    
    # Convert binary data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Unable to decode image")
    
    # Perform OCR
    result = ocr_instance.ocr(image, cls=True)
    lines = result[0] if result else []
    
    recognized_items = []
    recognized_words = []
    
    # Process results
    for line in lines:
        box = np.array(line[0]).astype(np.int32)
        text = str(line[1][0]).strip()
        confidence = float(line[1][1]) if line[1] and len(line[1]) > 1 else 0.0
        
        if text:
            recognized_items.append({
                "text": text,
                "confidence": confidence,
                "box": box.tolist(),
            })
            recognized_words.append(text)
            cv2.polylines(image, [box], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Save result image
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "result_with_boxes.jpg"), image)
    
    # Remove duplicates while preserving order
    unique_words = []
    seen = set()
    for word in recognized_words:
        normalized = word.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_words.append(normalized)
    
    return {
        "words": unique_words,
        "text": " ".join(unique_words).strip(),
        "items": recognized_items,
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "OCR Server is running"})


@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """
    OCR endpoint - accepts image as file upload
    
    Returns:
        JSON with recognized words, text, and items
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "errCode": 1,
                "errMessage": "Missing image file",
                "words": [],
                "text": "",
                "items": []
            }), 400
        
        image_file = request.files['image']
        
        if not image_file or image_file.filename == '':
            return jsonify({
                "errCode": 1,
                "errMessage": "Empty image file",
                "words": [],
                "text": "",
                "items": []
            }), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Process image
        payload = recognize_image_data(image_data)
        
        return jsonify({
            "errCode": 0,
            "errMessage": "OK",
            "words": payload.get("words", []),
            "text": payload.get("text", ""),
            "items": payload.get("items", []),
        })
    
    except Exception as e:
        print(f"[OCR Server] Error: {str(e)}", flush=True)
        return jsonify({
            "errCode": -1,
            "errMessage": f"Internal server error: {str(e)}",
            "words": [],
            "text": "",
            "items": []
        }), 500


@app.route('/ocr-base64', methods=['POST'])
def ocr_base64_endpoint():
    """
    OCR endpoint - accepts image as base64
    
    Payload: {"image": "base64-encoded-image"}
    
    Returns:
        JSON with recognized words, text, and items
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "errCode": 1,
                "errMessage": "Missing image data",
                "words": [],
                "text": "",
                "items": []
            }), 400
        
        import base64
        image_base64 = data['image']
        
        # Decode base64 to binary
        image_data = base64.b64decode(image_base64)
        
        # Process image
        payload = recognize_image_data(image_data)
        
        return jsonify({
            "errCode": 0,
            "errMessage": "OK",
            "words": payload.get("words", []),
            "text": payload.get("text", ""),
            "items": payload.get("items", []),
        })
    
    except Exception as e:
        print(f"[OCR Server] Error: {str(e)}", flush=True)
        return jsonify({
            "errCode": -1,
            "errMessage": f"Internal server error: {str(e)}",
            "words": [],
            "text": "",
            "items": []
        }), 500


if __name__ == '__main__':
    # Initialize model before starting server
    initialize_ocr()
    
    # Get port from environment or use default
    port = int(os.environ.get('OCR_SERVER_PORT', 5001))
    
    print(f"[OCR Server] Starting OCR server on port {port}...", flush=True)
    app.run(host='127.0.0.1', port=port, debug=False)
