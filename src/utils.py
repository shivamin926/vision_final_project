import math
import os

import cv2
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_mask(mask, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def largest_contour_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)

    if not contours:
        return new_mask

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(new_mask, [largest], -1, 255, thickness=-1)
    return new_mask


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0.0

    return (4 * math.pi * area) / (perimeter ** 2)


def contour_mask(shape, contour):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def contour_features(contour, image_shape):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    bbox_area = float(w * h) if w and h else 0.0

    return {
        "bbox": (x, y, w, h),
        "area": area,
        "aspect_ratio": max(w / float(h), h / float(w)) if w and h else 0.0,
        "circularity": circularity(contour),
        "solidity": area / hull_area if hull_area else 0.0,
        "extent": area / bbox_area if bbox_area else 0.0,
        "touches_edge": touches_image_edge((x, y, w, h), image_shape),
    }


def touches_image_edge(bbox, image_shape, margin=4):
    x, y, w, h = bbox
    height, width = image_shape[:2]
    return (
        x <= margin
        or y <= margin
        or x + w >= width - margin
        or y + h >= height - margin
    )


def hue_distance(value, reference):
    diff = abs(float(value) - float(reference))
    return min(diff, 180.0 - diff)


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = (aw * ah) + (bw * bh) - inter_area
    return inter_area / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.35):
    kept = []

    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        overlaps = [
            bbox_iou(detection["bbox"], kept_detection["bbox"])
            for kept_detection in kept
            if detection["label"] == kept_detection["label"]
        ]

        if overlaps and max(overlaps) >= iou_threshold:
            continue

        kept.append(detection)

    return kept


def save_rgba(bgr_image, alpha_mask, save_path):
    b, g, r = cv2.split(bgr_image)
    rgba = cv2.merge([b, g, r, alpha_mask])
    cv2.imwrite(save_path, rgba)
