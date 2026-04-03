# Multi-Class Fruit Detection and Labelling

This project detects and labels multiple fruit classes using classical computer vision instead of a trained neural network. The pipeline is built around HSV colour segmentation, morphological cleanup, contour extraction, geometric feature analysis, and interpretable rule-based classification.

## Supported classes

- apple
- banana
- orange

The detector is designed so a new fruit class can be added later by extending the HSV ranges and template references.

## Method

1. Convert each image from BGR to HSV.
2. Build colour masks for apple-like red, banana-like yellow, and lime-like green regions.
3. Build a foreground mask by removing bright low-saturation background pixels.
4. Clean masks with morphological opening and closing.
5. Extract connected components / contours as fruit candidates.
6. Compute interpretable features for each candidate:
   - bounding box
   - contour area
   - aspect ratio
   - circularity
   - solidity
   - dominant hue
   - colour coverage ratios for each fruit class
7. Compare each candidate against reference statistics measured from the template dataset in `data/templates/`.
8. Apply duplicate suppression to remove overlapping detections.
9. Save annotated images, diagnostic masks, and a prediction CSV.

## Project structure

- `src/detect_fruits.py`: main detector and prediction export
- `src/evaluate.py`: IoU-based evaluation with per-class precision/recall/F1
- `src/bootstrap_ground_truth.py`: seeds `labels.csv` from predictions for manual correction
- `src/extract_templates.py`: removes white template backgrounds and saves RGBA fruit templates
- `src/utils.py`: shared geometry, morphology, IoU, and utility helpers
- `data/test_images/`: test images
- `data/templates/`: fruit templates used to derive class reference features
- `data/ground_truth/labels.csv`: manual ground truth boxes for evaluation
- `output/detections/`: annotated output images
- `output/masks/`: diagnostic masks
- `output/metrics/`: predictions and evaluation tables

## Installation

```bash
pip install -r requirements.txt
```

## Run detection

```bash
python src/detect_fruits.py
```

Useful options:

```bash
python src/detect_fruits.py --keep-unknown
python src/detect_fruits.py --min-area-ratio 0.0015
```

Outputs:

- annotated images in `output/detections/`
- diagnostic masks in `output/masks/`
- detection table in `output/metrics/predictions.csv`

## Add ground truth and evaluate

Ground-truth CSV format:

```csv
image,label,x,y,w,h
1.jpg,banana,324,285,383,699
1.jpg,apple,672,274,271,308
```

If you do not have manual labels yet, create a reviewable starting point with:

```bash
python src/bootstrap_ground_truth.py
```

Then review `data/ground_truth/labels.csv`, correct any mistakes, and run:

```bash
python src/evaluate.py
```

Outputs:

- `output/metrics/evaluation_summary.csv`
- `output/metrics/matches.csv`

## Notes for report / viva

Why HSV?
Colour separation is more stable in HSV than raw RGB because hue is less sensitive to illumination than direct channel intensities.

Why rule-based classification?
The system stays transparent and explainable. Each label comes from visible colour evidence plus shape features such as circularity and aspect ratio.

Typical failure cases:

- overlapping fruits merging into a single contour
- heavy shadows changing hue and saturation
- fruit touching image borders
- cluttered backgrounds with fruit-like colours
- yellow fruits with similar shape being confused without an extra class-specific rule

