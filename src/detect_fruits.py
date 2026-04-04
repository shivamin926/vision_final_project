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
    "apple":   (0,   0, 255),
    "banana":  (0, 215, 255),
    "orange":  (0, 165, 255),
    "unknown": (255, 255, 255),
}

# HSV ranges — tightly separated so apple/orange/banana don't bleed into each other.
#   apple  : two red arcs only (0-12 dark red, 165-180 pink/magenta), sat >= 70
#   banana : pure yellow 20-34, sat >= 80
#   orange : narrow band 10-22, sat >= 110  (sits between banana yellow and apple red)
COLOR_RANGES = {
    "apple": [
        (np.array([  0,  70,  40], dtype=np.uint8), np.array([ 12, 255, 255], dtype=np.uint8)),
        (np.array([165,  60,  40], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    ],
    "banana": [
        (np.array([20, 80, 80], dtype=np.uint8), np.array([34, 255, 255], dtype=np.uint8)),
    ],
    "orange": [
        (np.array([10, 110, 80], dtype=np.uint8), np.array([22, 255, 255], dtype=np.uint8)),
    ],
}

# White background mask (studio / paper)
WHITE_LOWER = np.array([  0,   0, 150], dtype=np.uint8)
WHITE_UPPER = np.array([180,  60, 255], dtype=np.uint8)

# Dark/grey background mask (wooden table, dark backdrops).
# Pixels with low saturation AND low-to-mid value are background, not fruit.
DARK_BG_LOWER = np.array([  0,   0,   0], dtype=np.uint8)
DARK_BG_UPPER = np.array([180,  50,  90], dtype=np.uint8)

# Any bounding box that covers more than this fraction of the image is rejected.
# Kills the "whole scene" banana boxes seen in images 4 and 7.
MAX_BBOX_IMAGE_RATIO = 0.50

# Minimum contour area as a fraction of image area
DEFAULT_MIN_AREA_RATIO = 0.003


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and label multiple fruit classes.")
    parser.add_argument("--input-dir",       default=TEST_DIR)
    parser.add_argument("--output-dir",      default=OUTPUT_DIR)
    parser.add_argument("--mask-dir",        default=MASK_DIR)
    parser.add_argument("--prediction-file", default=PREDICTION_FILE)
    parser.add_argument("--min-area-ratio",  type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--keep-unknown",    action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def build_color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def build_color_masks(hsv):
    masks = {}
    for label, ranges in COLOR_RANGES.items():
        m = build_color_mask(hsv, ranges)
        k = 9 if label == "banana" else 5
        masks[label] = clean_mask(m, kernel_size=k)
    return masks


def build_foreground_mask(hsv):
    """
    Foreground = everything that is NOT a known background colour.
    We subtract both white/light backgrounds AND dark/grey backgrounds so the
    detector works correctly on dark-table images (images 2, 8) without
    treating the entire frame as foreground.
    """
    white_mask  = cv2.inRange(hsv, WHITE_LOWER,  WHITE_UPPER)
    dark_mask   = cv2.inRange(hsv, DARK_BG_LOWER, DARK_BG_UPPER)
    bg_mask     = cv2.bitwise_or(white_mask, dark_mask)
    fg          = cv2.bitwise_not(bg_mask)
    fg          = clean_mask(fg, kernel_size=5, iterations=2)
    return fg


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def contour_to_feature_vector(contour, bgr_image, hsv_image, color_masks):
    region_mask = contour_mask(bgr_image.shape, contour)
    features    = contour_features(contour, bgr_image.shape)
    features["pixel_count"] = int(cv2.countNonZero(region_mask))

    mean_hsv = cv2.mean(hsv_image, mask=region_mask)[:3]
    features["mean_h"] = float(mean_hsv[0])
    features["mean_s"] = float(mean_hsv[1])
    features["mean_v"] = float(mean_hsv[2])

    hue_hist = cv2.calcHist([hsv_image], [0], region_mask, [180], [0, 180]).flatten()
    features["dominant_hue"] = float(np.argmax(hue_hist)) if hue_hist.size else 0.0

    for label, mask in color_masks.items():
        overlap = cv2.bitwise_and(mask, mask, mask=region_mask)
        ratio   = cv2.countNonZero(overlap) / float(max(features["pixel_count"], 1))
        features[f"{label}_ratio"] = float(ratio)

    return features


# ---------------------------------------------------------------------------
# Template references
# ---------------------------------------------------------------------------

def load_template_references(template_dir=TEMPLATE_DIR):
    references = {}
    for fruit in FRUIT_ORDER:
        fruit_dir = os.path.join(template_dir, fruit)
        values    = []
        if not os.path.isdir(fruit_dir):
            continue
        for filename in os.listdir(fruit_dir):
            if not filename.lower().endswith(".png"):
                continue
            path = os.path.join(fruit_dir, filename)
            rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[-1] < 4:
                continue
            bgr   = rgba[:, :, :3]
            alpha = rgba[:, :,  3]
            contours, _ = cv2.findContours(
                (alpha > 0).astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour     = max(contours, key=cv2.contourArea)
            hsv         = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            color_masks = build_color_masks(hsv)
            values.append(contour_to_feature_vector(contour, bgr, hsv, color_masks))

        if not values:
            continue
        references[fruit] = {
            "dominant_hue": median(v["dominant_hue"]         for v in values),
            "aspect_ratio": median(v["aspect_ratio"]         for v in values),
            "circularity":  median(v["circularity"]          for v in values),
            "solidity":     median(v["solidity"]             for v in values),
            "apple_ratio":  median(v.get("apple_ratio",  0.) for v in values),
            "banana_ratio": median(v.get("banana_ratio", 0.) for v in values),
            "orange_ratio": median(v.get("orange_ratio", 0.) for v in values),
        }
    return references


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_against_reference(features, fruit, references):
    ref = references.get(fruit)
    if not ref:
        return -1.0

    color_key   = f"{fruit}_ratio"
    color_ratio = features.get(color_key, 0.0)

    hue_score    = 1.0 - min(hue_distance(features["dominant_hue"], ref["dominant_hue"]) / 20.0, 1.0)
    aspect_score = 1.0 - min(abs(features["aspect_ratio"] - ref["aspect_ratio"]) / 1.5,   1.0)
    circ_score   = 1.0 - min(abs(features["circularity"]  - ref["circularity"])  / 0.6,   1.0)
    solid_score  = 1.0 - min(abs(features["solidity"]     - ref["solidity"])     / 0.5,   1.0)
    color_score  = min(color_ratio / max(ref[color_key], 0.20), 1.0)
    sat_score    = min(features["mean_s"] / 160.0, 1.0)

    #               hue    asp    circ   solid  color  sat
    WEIGHTS = {
        "apple":  (0.25, 0.07, 0.16, 0.12, 0.30, 0.10),
        "banana": (0.10, 0.25, 0.08, 0.10, 0.37, 0.10),
        "orange": (0.22, 0.07, 0.22, 0.10, 0.29, 0.10),
    }[fruit]

    return (
          hue_score    * WEIGHTS[0]
        + aspect_score * WEIGHTS[1]
        + circ_score   * WEIGHTS[2]
        + solid_score  * WEIGHTS[3]
        + color_score  * WEIGHTS[4]
        + sat_score    * WEIGHTS[5]
    )


# ---------------------------------------------------------------------------
# Candidate classification
# ---------------------------------------------------------------------------

def candidate_labels(features):
    apple_ratio  = features["apple_ratio"]
    banana_ratio = features["banana_ratio"]
    orange_ratio = features["orange_ratio"]
    mean_s       = features["mean_s"]
    labels       = []

    if apple_ratio  >= 0.20 and features["solidity"]     >= 0.68 and features["circularity"] >= 0.36 and mean_s >= 55:
        labels.append("apple")
    if banana_ratio >= 0.16 and features["aspect_ratio"] >= 1.40 and features["solidity"]    >= 0.50 and mean_s >= 65:
        labels.append("banana")
    if orange_ratio >= 0.16 and features["circularity"]  >= 0.40 and features["solidity"]    >= 0.60 and mean_s >= 75:
        labels.append("orange")

    # When multiple classes qualify, keep only the strongest colour match.
    if len(labels) > 1:
        ratios = {"apple": apple_ratio, "banana": banana_ratio, "orange": orange_ratio}
        labels = [max(labels, key=lambda l: ratios[l])]

    return labels


def classify_candidate(features, references):
    labels = candidate_labels(features)
    if not labels:
        return "unknown", 0.0

    scores = {f: score_against_reference(features, f, references) for f in labels}
    label, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score < 0.50:
        return "unknown", max(best_score, 0.0)
    return label, best_score


# ---------------------------------------------------------------------------
# Candidate mask generation
# ---------------------------------------------------------------------------

def generate_candidate_masks(hsv, color_masks):
    foreground_mask = build_foreground_mask(hsv)
    union_mask = np.zeros_like(foreground_mask)
    for mask in color_masks.values():
        union_mask = cv2.bitwise_or(union_mask, mask)

    # Only merge foreground into candidate mask when foreground is sparse
    # (i.e. a clean white-background image).  On dark/busy backgrounds the
    # foreground mask is large and merging it creates a single giant blob that
    # swallows the whole scene.  Lowering the threshold from 0.18 → 0.08
    # keeps the behaviour correct for white-bg images while preventing the
    # dark-bg flood.
    non_white_ratio = cv2.countNonZero(foreground_mask) / float(foreground_mask.size)
    if non_white_ratio < 0.08:
        candidate_mask = cv2.bitwise_or(union_mask, foreground_mask)
    else:
        candidate_mask = union_mask

    candidate_mask = clean_mask(candidate_mask, kernel_size=5, iterations=2)
    return foreground_mask, union_mask, candidate_mask


# ---------------------------------------------------------------------------
# Contour collection
# ---------------------------------------------------------------------------

def collect_candidate_contours(masks, image_area, min_area):
    candidates       = []
    min_area_factors = {"apple": 0.35, "banana": 0.25, "orange": 0.30}

    for source_name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        src_min = (
            image_area * min_area_factors.get(source_name, 0.3) * 0.0015
            if source_name in FRUIT_ORDER
            else min_area
        )
        for contour in contours:
            if cv2.contourArea(contour) < src_min:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # Reject contours whose bounding box covers too much of the image.
            # This prevents a single large colour blob from generating a box
            # around the entire scene (was 0.93 — far too permissive).
            if (w * h) / float(image_area) > MAX_BBOX_IMAGE_RATIO:
                continue
            candidates.append((source_name, contour))

    return candidates


# ---------------------------------------------------------------------------
# Per-detection reliability filter
# ---------------------------------------------------------------------------

def is_reliable_detection(det):
    area  = det["area"]
    label = det["label"]
    asp   = det["aspect_ratio"]
    circ  = det["circularity"]

    if asp  > 5.0:  return False
    if circ < 0.15: return False
    if area < 500:  return False   # raised from 100/400 — kills tiny phantom boxes

    if label == "apple":
        return (
            det["score"]       >= 0.50
            and det["apple_ratio"] >= 0.18
            and asp                <= 1.90
            and det["solidity"]   >= 0.65
            and (circ >= 0.40 or area >= 12000)
        )
    if label == "banana":
        return (
            det["score"]        >= 0.50
            and det["banana_ratio"] >= 0.14
            and asp                 >= 1.30
            and det["solidity"]    >= 0.48
            and circ                >= 0.15
        )
    if label == "orange":
        return (
            det["score"]        >= 0.50
            and det["orange_ratio"] >= 0.14
            and asp                 <= 2.20
            and det["solidity"]    >= 0.60
            and circ                >= 0.38
        )
    return False


# ---------------------------------------------------------------------------
# Suppression helpers
# ---------------------------------------------------------------------------

def suppress_conflicts(detections, iou_threshold=0.55):
    """Cross-class suppression: if two detections of *different* classes
    overlap heavily, keep only the higher-scoring one."""
    kept = []
    for det in sorted(detections, key=lambda d: d["score"], reverse=True):
        if any(
            ex["label"] != det["label"]
            and bbox_iou(det["bbox"], ex["bbox"]) >= iou_threshold
            for ex in kept
        ):
            continue
        kept.append(det)
    return kept


def suppress_contained(detections):
    """Remove a detection if >=75% of its bbox area lies inside a
    higher-scoring detection of the SAME class.
    
    Standard IoU fails here because a small box *inside* a large box has
    low IoU (the union is large).  This containment check catches those
    phantom sub-boxes (images 1, 8)."""
    kept   = []
    ranked = sorted(detections, key=lambda d: d["score"], reverse=True)
    for det in ranked:
        dx, dy, dw, dh = det["bbox"]
        det_area = dw * dh
        if det_area == 0:
            continue
        absorbed = False
        for ex in kept:
            if ex["label"] != det["label"]:
                continue
            ex_x, ex_y, ex_w, ex_h = ex["bbox"]
            ix  = max(dx, ex_x);          iy  = max(dy, ex_y)
            ix2 = min(dx + dw, ex_x + ex_w); iy2 = min(dy + dh, ex_y + ex_h)
            if ix2 <= ix or iy2 <= iy:
                continue
            inter = (ix2 - ix) * (iy2 - iy)
            if inter / float(det_area) >= 0.75:
                absorbed = True
                break
        if not absorbed:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def detect_image(image, references, min_area_ratio):
    hsv         = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_masks = build_color_masks(hsv)
    foreground_mask, union_mask, candidate_mask = generate_candidate_masks(hsv, color_masks)

    image_area = image.shape[0] * image.shape[1]
    min_area   = max(500.0, image_area * min_area_ratio)

    search_masks = {
        "foreground": foreground_mask,
        "candidate":  candidate_mask,
        **color_masks,
    }

    detections = []
    for source_name, contour in collect_candidate_contours(search_masks, image_area, min_area):
        features = contour_to_feature_vector(contour, image, hsv, color_masks)
        features["source"] = source_name

        if source_name in FRUIT_ORDER and features[f"{source_name}_ratio"] < 0.10:
            continue

        label, score = classify_candidate(features, references)
        if label == "unknown":
            continue

        features["label"] = label
        features["score"] = float(score)
        if is_reliable_detection(features):
            detections.append(features)

    # Pass 1: class-aware NMS (same label, IoU-based)
    detections = non_max_suppression(detections, iou_threshold=0.30)
    # Pass 2: cross-class conflict removal
    detections = suppress_conflicts(detections, iou_threshold=0.55)
    # Pass 3: remove small boxes fully contained within a larger same-class box
    detections = suppress_contained(detections)
    detections.sort(key=lambda d: d["score"], reverse=True)

    diagnostics = {
        "foreground": foreground_mask,
        "candidate":  candidate_mask,
        **color_masks,
        "union": union_mask,
    }
    return detections, diagnostics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_detections(image, detections):
    output = image.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        color = CLASS_COLORS.get(label, CLASS_COLORS["unknown"])
        text  = f"{label} {det['score']:.2f}"
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, text, (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_predictions(prediction_file, rows):
    ensure_dir(os.path.dirname(prediction_file))
    fieldnames = [
        "image", "label", "score",
        "x", "y", "w", "h",
        "area", "aspect_ratio", "circularity", "solidity",
        "apple_ratio", "banana_ratio", "orange_ratio",
        "dominant_hue", "touches_edge", "source",
    ]
    with open(prediction_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.mask_dir)
    ensure_dir(os.path.dirname(args.prediction_file))

    references      = load_template_references()
    prediction_rows = []

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if "Zone.Identifier" in filename:
            continue

        path  = os.path.join(args.input_dir, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        detections, diagnostics = detect_image(image, references, args.min_area_ratio)
        filtered  = detections if args.keep_unknown else [d for d in detections if d["label"] != "unknown"]
        annotated = draw_detections(image, filtered)
        cv2.imwrite(os.path.join(args.output_dir, filename), annotated)

        for mask_name, mask in diagnostics.items():
            cv2.imwrite(os.path.join(args.mask_dir, f"{mask_name}_{filename}"), mask)

        for det in filtered:
            x, y, w, h = det["bbox"]
            prediction_rows.append({
                "image":        filename,
                "label":        det["label"],
                "score":        round(det["score"],        4),
                "x": x, "y": y, "w": w, "h": h,
                "area":         round(det["area"],         2),
                "aspect_ratio": round(det["aspect_ratio"], 4),
                "circularity":  round(det["circularity"],  4),
                "solidity":     round(det["solidity"],     4),
                "apple_ratio":  round(det["apple_ratio"],  4),
                "banana_ratio": round(det["banana_ratio"], 4),
                "orange_ratio": round(det["orange_ratio"], 4),
                "dominant_hue": round(det["dominant_hue"], 2),
                "touches_edge": int(det["touches_edge"]),
                "source":       det["source"],
            })

        print(f"[INFO] {filename}: {len(filtered)} detection(s)")
        for det in filtered:
            print(
                f"  {det['label']:<6} score={det['score']:.2f}  bbox={det['bbox']}  "
                f"asp={det['aspect_ratio']:.2f}  circ={det['circularity']:.2f}  src={det['source']}"
            )

    write_predictions(args.prediction_file, prediction_rows)
    print(f"[DONE] Predictions → {args.prediction_file}")


if __name__ == "__main__":
    main()