# Text Region Detection

**Computer Vision Team Project — Theme #17**
Detect areas containing text in images (pre-OCR pipeline).

## Team roles

| Person | Role | Pipeline stages | Files |
|--------|------|-----------------|-------|
| Person 1 | Image Processing Specialist | Enhance + Segment | `src/preprocessing/` |
| Person 2 | Morphology & Report Lead | Clean + Visualise | `src/morphology/`, `src/visualization/` |
| Person 3 | Lead CV Engineer | Detect + Decide + Integration | `src/detection/`, `main.py` |

## Pipeline

```
image → enhance → segment → clean → detect → decide
         (P1)      (P1)      (P2)    (P3)     (P3)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Single image
python main.py --input data/sample_images/sample1.png

# Whole folder
python main.py --input data/sample_images/ --output output/

# Tune parameters
python main.py --input data/sample_images/ \
               --strategy adaptive \
               --min-area 150 \
               --max-aspect 15.0
```

## Outputs (per image)

```
output/<image_name>/
    original.png        original input image
    enhanced.png        grayscale, CLAHE + gamma corrected
    seg_mask.png        binary segmentation mask
    cleaned_mask.png    morphologically cleaned mask
    detection.png       original + green bounding boxes
    stage_grid.png      2x3 panel overview of all stages
    decision.json       structured result
output/
    contact_sheet.png   thumbnail grid of all detection results
    summary.json        one-line result per image
```

## Tests

```bash
pytest tests/ -v
```

## Interface contracts

```python
# Person 1 outputs
enhanced_img : np.ndarray  # shape (H, W), uint8, grayscale
seg_mask     : np.ndarray  # shape (H, W), uint8, {0, 255}

# Person 2 outputs
cleaned_mask : np.ndarray  # shape (H, W), uint8, {0, 255}

# Person 3 outputs
boxes    : list[tuple[int, int, int, int]]  # [(x, y, w, h), ...]
decision : dict  # {text_found, region_count, coverage, regions, verdict}
```
