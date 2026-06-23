import json
import sys
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


BASE_DIR = Path(__file__).resolve().parent
DET_MODEL_DIR = BASE_DIR / "inference" / "det_mv3_db"
OUTPUT_DIR = BASE_DIR / "output"


def recognize_image(image_path):
    ocr = PaddleOCR(
        det_model_dir=str(DET_MODEL_DIR),
        use_angle_cls=True,
        lang="japan",
        show_log=False,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    result = ocr.ocr(str(image_path), cls=True)
    lines = result[0] if result else []

    recognized_items = []
    recognized_words = []

    for line in lines:
        box = np.array(line[0]).astype(np.int32)
        text = str(line[1][0]).strip()
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


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / "../Images/T2.png")
    payload = recognize_image(image_path)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()