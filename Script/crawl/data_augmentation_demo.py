import argparse
import json
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TEXT = "japanese sign menu"


def create_sample_image(width=640, height=480):
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (width - 20, height - 20), (10, 10, 10), 2)
    cv2.putText(
        image,
        "SAMPLE TEXT 123",
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    return image


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def adjust_saturation(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_crop_resize(image, rng, scale_min=0.7, scale_max=0.95):
    height, width = image.shape[:2]
    scale = rng.uniform(scale_min, scale_max)
    crop_h = max(1, int(height * scale))
    crop_w = max(1, int(width * scale))
    top = rng.randint(0, max(0, height - crop_h))
    left = rng.randint(0, max(0, width - crop_w))
    cropped = image[top : top + crop_h, left : left + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def add_noise(image, np_rng, sigma=10.0):
    noise = np_rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_random_image_aug(image, rng, np_rng):
    transforms = []

    angle = rng.choice([-15, -10, 10, 15])
    transforms.append(("rotation", lambda img: rotate_image(img, angle)))

    flip_code = rng.choice([1, 0, -1])
    transforms.append(("flip", lambda img: cv2.flip(img, flip_code)))

    alpha = rng.uniform(0.8, 1.2)
    beta = rng.randint(-30, 30)
    transforms.append(("brightness_contrast", lambda img: adjust_brightness_contrast(img, alpha, beta)))

    saturation = rng.uniform(0.7, 1.3)
    transforms.append(("saturation", lambda img: adjust_saturation(img, saturation)))

    transforms.append(("crop_resize", lambda img: random_crop_resize(img, rng)))

    sigma = rng.uniform(5.0, 20.0)
    transforms.append(("noise", lambda img: add_noise(img, np_rng, sigma)))

    rng.shuffle(transforms)
    selected = transforms[: rng.randint(2, 4)]

    output = image.copy()
    applied = []
    for name, func in selected:
        output = func(output)
        applied.append(name)
    return output, applied


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def synonym_replacement(tokens, rng, synonyms):
    candidates = [i for i, token in enumerate(tokens) if token in synonyms]
    if not candidates:
        return tokens
    index = rng.choice(candidates)
    replacement = rng.choice(synonyms[tokens[index]])
    new_tokens = tokens[:]
    new_tokens[index] = replacement
    return new_tokens


def random_deletion(tokens, rng, delete_prob=0.2):
    if len(tokens) <= 1:
        return tokens
    kept = [token for token in tokens if rng.random() > delete_prob]
    return kept if kept else tokens


def random_swap(tokens, rng):
    if len(tokens) < 2:
        return tokens
    i, j = rng.sample(range(len(tokens)), 2)
    new_tokens = tokens[:]
    new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
    return new_tokens


def random_insertion(tokens, rng, synonyms):
    candidates = [token for token in tokens if token in synonyms]
    if not candidates:
        return tokens
    token = rng.choice(candidates)
    insert_word = rng.choice(synonyms[token])
    position = rng.randint(0, len(tokens))
    new_tokens = tokens[:]
    new_tokens.insert(position, insert_word)
    return new_tokens


def augment_text(text, rng):
    synonyms = {
        "japanese": ["jp"],
        "sign": ["board", "notice"],
        "menu": ["list", "catalog"],
        "book": ["text"],
        "street": ["road"],
    }

    normalized = normalize_text(text)
    tokens = normalized.split()
    if not tokens:
        return [normalized]

    operations = [
        lambda t: synonym_replacement(t, rng, synonyms),
        lambda t: random_deletion(t, rng),
        lambda t: random_swap(t, rng),
        lambda t: random_insertion(t, rng, synonyms),
    ]

    rng.shuffle(operations)
    steps = rng.randint(1, 3)
    for op in operations[:steps]:
        tokens = op(tokens)

    return [" ".join(tokens).strip()]


def main():
    parser = argparse.ArgumentParser(description="Simple data augmentation demo")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--num-variants", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output" / "augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path:
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {args.image_path}")
    else:
        image = create_sample_image()

    base_image_path = output_dir / "base_image.jpg"
    cv2.imwrite(str(base_image_path), image)

    augmented_images = []
    for index in range(args.num_variants):
        aug_image, applied = apply_random_image_aug(image, rng, np_rng)
        out_path = output_dir / f"image_aug_{index + 1:02d}.jpg"
        cv2.imwrite(str(out_path), aug_image)
        augmented_images.append({"path": str(out_path), "applied": applied})

    text_variants = []
    for _ in range(args.num_variants):
        text_variants.extend(augment_text(args.text, rng))

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_image": str(base_image_path),
        "augmented_images": augmented_images,
        "text_variants": text_variants,
    }

    meta_path = output_dir / "augmentation_meta.json"
    with open(meta_path, "w", encoding="utf-8") as output:
        json.dump(meta, output, ensure_ascii=False, indent=2)

    print(f"Saved {len(augmented_images)} augmented images to {output_dir}")
    print(f"Saved {len(text_variants)} text variants to {meta_path}")


if __name__ == "__main__":
    main()
