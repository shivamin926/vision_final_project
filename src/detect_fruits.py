import argparse
import csv
import os
from statistics import median

import cv2
import numpy as np

from utils import bbox_iou, clean_mask, contour_features, contour_mask, ensure_dir, hue_distance, non_max_suppression

TEST_DIR = "data/test_images"
TEMPLATE_DIR = "data/templates"
OUTPUT_DIR = "output/detections"
MASK_DIR = "output/masks"
METRICS_DIR = "output/metrics"
PREDICTION_FILE = os.path.join(METRICS_DIR, "predictions.csv")

FRUIT_ORDER = ["apple", "banana", "orange"]
CLASS_COLORS = {
    "apple": (0, 0, 255),
    "banana": (0, 215, 255),
    "orange": (0, 165, 255),
    "unknown": (255, 255, 255),
}
COLOR_RANGES = {
    "apple": [
        (np.array([0, 65, 45], dtype=np.uint8), np.array([24, 255, 255], dtype=np.uint8)),
        (np.array([170, 65, 45], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    ],
    "banana": [
        (np.array([18, 70, 70], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
    ],
    "orange": [
        (np.array([10, 50, 50], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
    ],
}
WHITE_LOWER = np.array([0, 0, 140], dtype=np.uint8)
WHITE_UPPER = np.array([180, 95, 255], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and label multiple fruit classes.")
    parser.add_argument("--input-dir", default=TEST_DIR, help="Directory containing test images.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for annotated detections.")
    parser.add_argument("--mask-dir", default=MASK_DIR, help="Directory for diagnostic masks.")
    parser.add_argument("--prediction-file", default=PREDICTION_FILE, help="CSV path for saved predictions.")
    parser.add_argument("--min-area-ratio", type=float, default=0.0018, help="Minimum contour area as a ratio of image area.")
    parser.add_argument("--keep-unknown", action="store_true", help="Keep detections that do not match a fruit class.")
    return parser.parse_args()


def build_color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def build_color_masks(hsv):
    masks = {label: build_color_mask(hsv, ranges) for label, ranges in COLOR_RANGES.items()}
    for label, mask in masks.items():
        kernel_size = 7 if label == "banana" else 5
        masks[label] = clean_mask(mask, kernel_size=kernel_size)
    return masks


def build_foreground_mask(hsv):
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    foreground_mask = cv2.bitwise_not(white_mask)
    foreground_mask = clean_mask(foreground_mask, kernel_size=5, iterations=2)
    return foreground_mask


def contour_to_feature_vector(contour, bgr_image, hsv_image, color_masks):
    region_mask = contour_mask(bgr_image.shape, contour)
    features = contour_features(contour, bgr_image.shape)
    features["pixel_count"] = int(cv2.countNonZero(region_mask))

    mean_hsv = cv2.mean(hsv_image, mask=region_mask)[:3]
    features["mean_h"] = float(mean_hsv[0])
    features["mean_s"] = float(mean_hsv[1])
    features["mean_v"] = float(mean_hsv[2])

    hue_hist = cv2.calcHist([hsv_image], [0], region_mask, [180], [0, 180]).flatten()
    features["dominant_hue"] = float(np.argmax(hue_hist)) if hue_hist.size else 0.0

    for label, mask in color_masks.items():
        overlap = cv2.bitwise_and(mask, mask, mask=region_mask)
        ratio = cv2.countNonZero(overlap) / float(max(features["pixel_count"], 1))
        features[f"{label}_ratio"] = float(ratio)

    return features


def load_template_references(template_dir=TEMPLATE_DIR):
    references = {}

    for fruit in FRUIT_ORDER:
        fruit_dir = os.path.join(template_dir, fruit)
        values = []
        if not os.path.isdir(fruit_dir):
            continue

        for filename in os.listdir(fruit_dir):
            if not filename.lower().endswith(".png"):
                continue

            path = os.path.join(fruit_dir, filename)
            rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[-1] < 4:
                continue

            bgr = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
            contours, _ = cv2.findContours((alpha > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            color_masks = build_color_masks(hsv)
            values.append(contour_to_feature_vector(contour, bgr, hsv, color_masks))

        if not values:
            continue

        references[fruit] = {
            "dominant_hue": median(item["dominant_hue"] for item in values),
            "aspect_ratio": median(item["aspect_ratio"] for item in values),
            "circularity": median(item["circularity"] for item in values),
            "solidity": median(item["solidity"] for item in values),
            "apple_ratio": median(item.get("apple_ratio", 0.0) for item in values),
            "banana_ratio": median(item.get("banana_ratio", 0.0) for item in values),
            "orange_ratio": median(item.get("orange_ratio", 0.0) for item in values),
        }

    return references


def score_against_reference(features, fruit, references):
    reference = references.get(fruit)
    if not reference:
        return -1.0

    color_key = f"{fruit}_ratio"
    color_ratio = features.get(color_key, 0.0)

    hue_score = 1.0 - min(hue_distance(features["dominant_hue"], reference["dominant_hue"]) / 30.0, 1.0)
    aspect_score = 1.0 - min(abs(features["aspect_ratio"] - reference["aspect_ratio"]) / 1.8, 1.0)
    circ_score = 1.0 - min(abs(features["circularity"] - reference["circularity"]) / 0.7, 1.0)
    solid_score = 1.0 - min(abs(features["solidity"] - reference["solidity"]) / 0.55, 1.0)
    color_score = min(color_ratio / max(reference[color_key], 0.2), 1.0)

    weights = {
        "apple": (0.28, 0.10, 0.24, 0.18, 0.20),
        "banana": (0.08, 0.30, 0.12, 0.15, 0.35),
        "orange": (0.22, 0.10, 0.28, 0.15, 0.25),
    }[fruit]

    return (
        hue_score * weights[0]
        + aspect_score * weights[1]
        + circ_score * weights[2]
        + solid_score * weights[3]
        + color_score * weights[4]
    )


def candidate_labels(features):
    apple_ratio = features["apple_ratio"]
    banana_ratio = features["banana_ratio"]
    orange_ratio = features["orange_ratio"]
    labels = []

    if apple_ratio >= 0.18 and features["solidity"] >= 0.68 and features["circularity"] >= 0.30:
        labels.append("apple")
    if banana_ratio >= 0.14 and features["aspect_ratio"] >= 1.4 and features["solidity"] >= 0.4:
        labels.append("banana")
    if orange_ratio >= 0.12 and features["circularity"] >= 0.38 and features["solidity"] >= 0.58:
        labels.append("orange")

    # Resolve very mixed regions by favouring the strongest colour explanation.
    if len(labels) > 1:
        ratios = {"apple": apple_ratio, "banana": banana_ratio, "orange": orange_ratio}
        strongest = max(labels, key=lambda label: ratios[label])
        labels = [strongest] + [label for label in labels if label != strongest and ratios[label] >= ratios[strongest] * 0.85]

    return labels


def classify_candidate(features, references):
    labels = candidate_labels(features)
    if not labels:
        return "unknown", 0.0

    scores = {fruit: score_against_reference(features, fruit, references) for fruit in labels}
    label, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score < 0.45:
        return "unknown", max(best_score, 0.0)

    return label, best_score


def generate_candidate_masks(hsv, color_masks):
    foreground_mask = build_foreground_mask(hsv)
    union_mask = np.zeros_like(foreground_mask)
    for mask in color_masks.values():
        union_mask = cv2.bitwise_or(union_mask, mask)

    non_white_ratio = cv2.countNonZero(foreground_mask) / float(foreground_mask.size)
    if non_white_ratio >= 0.18:
        candidate_mask = cv2.bitwise_or(union_mask, foreground_mask)
    else:
        candidate_mask = union_mask

    candidate_mask = clean_mask(candidate_mask, kernel_size=5, iterations=2)
    return foreground_mask, union_mask, candidate_mask


def suppress_conflicts(detections, iou_threshold=0.6):
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(bbox_iou(detection["bbox"], existing["bbox"]) >= iou_threshold for existing in kept):
            continue
        kept.append(detection)
    return kept


def collect_candidate_contours(masks, image_area, min_area):
    candidates = []

    for source_name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        source_min_area = min_area * (0.45 if source_name in FRUIT_ORDER else 1.0)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < source_min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if (w * h) / float(image_area) > 0.93:
                continue

            candidates.append((source_name, contour))

    return candidates


def is_reliable_detection(det):
    area = det["area"]
    label = det["label"]

    if label == "apple":
        return det["score"] >= 0.72 and det["apple_ratio"] >= 0.18 and det["aspect_ratio"] <= 1.85 and (det["circularity"] >= 0.45 or area >= 15000)
    if label == "banana":
        return det["score"] >= 0.72 and det["banana_ratio"] >= 0.14 and det["aspect_ratio"] >= 1.45
    if label == "orange":
        return det["score"] >= 0.72 and det["orange_ratio"] >= 0.12 and det["aspect_ratio"] <= 2.1 and det["circularity"] >= 0.42
    return False


def detect_image(image, references, min_area_ratio):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_masks = build_color_masks(hsv)
    foreground_mask, union_mask, candidate_mask = generate_candidate_masks(hsv, color_masks)

    image_area = image.shape[0] * image.shape[1]
    min_area = max(120.0, image_area * min_area_ratio)
    search_masks = {
        "foreground": foreground_mask,
        "candidate": candidate_mask,
        **color_masks,
    }

    detections = []
    for source_name, contour in collect_candidate_contours(search_masks, image_area, min_area):
        features = contour_to_feature_vector(contour, image, hsv, color_masks)
        features["source"] = source_name

        if source_name in FRUIT_ORDER and features[f"{source_name}_ratio"] < 0.08:
            continue

        label, score = classify_candidate(features, references)
        if label == "unknown":
            continue

        features["label"] = label
        features["score"] = float(score)
        if is_reliable_detection(features):
            detections.append(features)

    detections = non_max_suppression(detections, iou_threshold=0.3)
    detections = suppress_conflicts(detections, iou_threshold=0.65)
    detections.sort(key=lambda item: item["score"], reverse=True)

    diagnostics = {
        "foreground": foreground_mask,
        "candidate": candidate_mask,
        **color_masks,
        "union": union_mask,
    }
    return detections, diagnostics


def draw_detections(image, detections):
    output = image.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        color = CLASS_COLORS.get(label, CLASS_COLORS["unknown"])
        text = f"{label} {det['score']:.2f}"

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, text, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output


def write_predictions(prediction_file, rows):
    ensure_dir(os.path.dirname(prediction_file))
    fieldnames = [
        "image",
        "label",
        "score",
        "x",
        "y",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "circularity",
        "solidity",
        "apple_ratio",
        "banana_ratio",
        "orange_ratio",
        "dominant_hue",
        "touches_edge",
        "source",
    ]

    with open(prediction_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.mask_dir)
    ensure_dir(os.path.dirname(args.prediction_file))

    references = load_template_references()
    prediction_rows = []

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if "Zone.Identifier" in filename:
            continue

        path = os.path.join(args.input_dir, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        detections, diagnostics = detect_image(image, references, args.min_area_ratio)
        filtered = detections if args.keep_unknown else [det for det in detections if det["label"] != "unknown"]
        annotated = draw_detections(image, filtered)
        cv2.imwrite(os.path.join(args.output_dir, filename), annotated)

        for mask_name, mask in diagnostics.items():
            cv2.imwrite(os.path.join(args.mask_dir, f"{mask_name}_{filename}"), mask)

        for det in filtered:
            x, y, w, h = det["bbox"]
            prediction_rows.append(
                {
                    "image": filename,
                    "label": det["label"],
                    "score": round(det["score"], 4),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": round(det["area"], 2),
                    "aspect_ratio": round(det["aspect_ratio"], 4),
                    "circularity": round(det["circularity"], 4),
                    "solidity": round(det["solidity"], 4),
                    "apple_ratio": round(det["apple_ratio"], 4),
                    "banana_ratio": round(det["banana_ratio"], 4),
                    "orange_ratio": round(det["orange_ratio"], 4),
                    "dominant_hue": round(det["dominant_hue"], 2),
                    "touches_edge": int(det["touches_edge"]),
                    "source": det["source"],
                }
            )

        print(f"[INFO] Processed {filename}: {len(filtered)} detections")
        for det in filtered:
            print(
                f"  - {det['label']:<6} score={det['score']:.2f} bbox={det['bbox']} "
                f"aspect={det['aspect_ratio']:.2f} circ={det['circularity']:.2f} source={det['source']}"
            )

    write_predictions(args.prediction_file, prediction_rows)
    print(f"[DONE] Saved predictions to {args.prediction_file}")


if __name__ == "__main__":
    main()
