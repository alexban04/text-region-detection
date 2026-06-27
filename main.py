"""
main.py — End-to-End Pipeline Runner
Person 3 — Lead CV Engineer

Usage:
    python main.py --input data/sample_images/sample1.png
    python main.py --input data/sample_images/ --output output/
    python main.py --input data/sample_images/ --min-area 150 --strategy adaptive
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from src.preprocessing import enhance, segment
from src.preprocessing.utils import iter_image_paths
from src.morphology import clean
from src.visualization import (
    draw_bounding_boxes,
    build_stage_grid,
    save_stage_grid,
    build_contact_sheet,
    save_contact_sheet,
)
from src.detection import find_text_boxes, make_decision, save_decision

def run_pipeline(
    image_path,
    output_dir,
    strategy="combined",
    min_area=200,
    max_aspect_ratio=20.0,
    nms_iou_threshold=0.3,
    panel_w=400,
    panel_h=300,
    verbose=True,
):
    """Run the full 5-stage pipeline on one image and save all outputs.

    Outputs saved to output_dir/<image_stem>/:
      original.png, enhanced.png, seg_mask.png, cleaned_mask.png,
      detection.png, stage_grid.png, decision.json
    """
    image_path = Path(image_path)
    stem = image_path.stem
    out = Path(output_dir) / stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    original, enhanced_img = enhance(image_path)
    cv2.imwrite(str(out / "original.png"), original)
    cv2.imwrite(str(out / "enhanced.png"), enhanced_img)
    if verbose:
        print(f" [1] Enhance   {time.perf_counter()-t0:.2f}s")

    seg_mask = segment(enhanced_img, strategy=strategy)
    cv2.imwrite(str(out / "seg_mask.png"), seg_mask)
    if verbose:
        print(f" [2] Segment {time.perf_counter()-t0:.2f}s")

    cleaned_mask = clean(seg_mask)
    cv2.imwrite(str(out / "cleaned_mask.png"), cleaned_mask)
    if verbose:
        print(f" [3] Clean {time.perf_counter()-t0:.2f}s")

    boxes = find_text_boxes(
        cleaned_mask,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        nms_iou_threshold=nms_iou_threshold,
    )
    detection_img = draw_bounding_boxes(original, boxes)
    cv2.imwrite(str(out / "detection.png"), detection_img)
    if verbose:
        print(f" [4] Detect {time.perf_counter()-t0:.2f}s  ({len(boxes)} box(es))")

    # Stage 5 — Decide
    decision = make_decision(boxes, original.shape)
    save_decision(decision, out / "decision.json")
    if verbose:
        print(f" [5] Decide {time.perf_counter()-t0:.2f}s  -> {decision['verdict']}")

    # Stage grid
    grid = build_stage_grid(
        original=original,
        enhanced=enhanced_img,
        seg_mask=seg_mask,
        cleaned_mask=cleaned_mask,
        detection_result=detection_img,
        decision=decision,
        panel_w=panel_w,
        panel_h=panel_h,
    )
    save_stage_grid(grid, out / "stage_grid.png")

    return {
        "decision": decision,
        "paths": {
            "original": out / "original.png",
            "enhanced": out / "enhanced.png",
            "seg_mask": out / "seg_mask.png",
            "cleaned_mask": out / "cleaned_mask.png",
            "detection": out / "detection.png",
            "stage_grid": out / "stage_grid.png",
            "decision_json": out / "decision.json",
        },
    }


def run_batch(input_path, output_dir, **pipeline_kwargs):
    """Run the pipeline on all images in a folder (or a single image)."""
    input_path = Path(input_path)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = list(iter_image_paths(input_path))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not image_paths:
        print(f"No supported images found in {input_path}", file=sys.stderr)
        return []

    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        entry = {"name": img_path.stem, "success": False, "error": None}
        try:
            result = run_pipeline(img_path, output_dir, **pipeline_kwargs)
            entry.update(result)
            entry["success"] = True
        except Exception as exc:
            entry["error"] = str(exc)
            print(f"  ERROR: {exc}", file=sys.stderr)
        results.append(entry)
    return results


def print_summary(results, output_dir):
    total = len(results)
    ok = sum(1 for r in results if r["success"])
    found = sum(1 for r in results if r["success"] and r["decision"]["text_found"])

    print(f"\n{'─'*50}")
    print(f"  Processed : {total} image(s)")
    print(f"  Success   : {ok}")
    print(f"  Text found: {found} / {ok}")
    print(f"  Output dir: {output_dir}")
    print(f"{'─'*50}\n")

    det_imgs, det_labels = [], []
    for r in results:
        if r["success"] and "paths" in r:
            img = cv2.imread(str(r["paths"]["detection"]))
            if img is not None:
                det_imgs.append(img)
                det_labels.append(r["name"])

    if det_imgs:
        sheet = build_contact_sheet(det_imgs, det_labels)
        sheet_path = Path(output_dir) / "contact_sheet.png"
        save_contact_sheet(sheet, sheet_path)
        print(f"  Contact sheet -> {sheet_path}")

    summary = [
        {
            "name":  r["name"],
            "success": r["success"],
            "verdict": r["decision"]["verdict"] if r["success"] else r["error"],
            "region_count": r["decision"]["region_count"] if r["success"] else 0,
            "coverage": r["decision"]["coverage"] if r["success"] else 0,
        }
        for r in results
    ]
    summary_path = Path(output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON  -> {summary_path}\n")


def build_parser():
    p = argparse.ArgumentParser(
        description="Text Region Detection — full CV pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="output/")
    p.add_argument("--strategy", choices=["adaptive", "gradient", "combined"], default="combined")
    p.add_argument("--min-area", type=int, default=200)
    p.add_argument("--max-aspect",type=float, default=20.0)
    p.add_argument("--nms-iou", type=float, default=0.3)
    p.add_argument("--panel-w", type=int, default=400)
    p.add_argument("--panel-h", type=int, default=300)
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    results = run_batch(
        input_path=args.input,
        output_dir=args.output,
        strategy=args.strategy,
        min_area=args.min_area,
        max_aspect_ratio=args.max_aspect,
        nms_iou_threshold=args.nms_iou,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
        verbose=not args.quiet,
    )
    print_summary(results, args.output)

if __name__ == "__main__":
    main()
