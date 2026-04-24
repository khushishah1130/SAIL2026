"""
Run volleyball analysis on a video and save annotated outputs.

Usage (from the src/ directory, with the venv activated):

    # Annotated videos only
    python run_video_analysis.py \
        --video_path videos/v1.mp4 \
        --output_dir results/run_001 \
        --mode both

    # Per-frame JSONL only (no video)
    python run_video_analysis.py \
        --video_path videos/v1.mp4 \
        --output_dir results/run_001 \
        --mode analysis

The JSONL file will be saved as: <output_dir>/frame_results.jsonl
Each line is one frame's data as a JSON object.
"""

import argparse
import json
from pathlib import Path

import cv2

from demo import run_object_detection, run_video_classification
from ml_manager import MLManager


def run_json_analysis(video_path: str, jsonl_path: str) -> None:
    """
    Run frame-by-frame analysis with MLManager and save results to a JSONL file.

    Each line in the JSONL file is a dict with:
      - frame_index
      - timestamp_seconds
      - ball_detection (single detection or null)
      - action_detections (list of detections)
      - player_keypoints (list, simplified as bboxes)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = float(fps)

    print("Initializing MLManager for JSON analysis...")
    manager = MLManager()

    frame_index = 0
    out_path = Path(jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect all: actions, single ball detection, player keypoints
            action_detections, ball_detection, player_keypoints = manager.detect_all(
                frame, conf_threshold=0.25, iou_threshold=0.45
            )

            record = {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps if fps > 0 else None,
                "ball_detection": None,
                "action_detections": [],
                "player_keypoints": [],
            }

            # Ball detection (single Detection object or None; detector may return list when empty)
            if ball_detection is not None and hasattr(ball_detection, "bbox"):
                b = ball_detection.bbox
                record["ball_detection"] = {
                    "bbox": {
                        "x1": float(b.x1),
                        "y1": float(b.y1),
                        "x2": float(b.x2),
                        "y2": float(b.y2),
                    },
                    "confidence": float(ball_detection.confidence),
                    "class_id": int(ball_detection.class_id),
                }

            # Actions
            for det in action_detections or []:
                b = det.bbox
                record["action_detections"].append(
                    {
                        "bbox": {"x1": float(b.x1), "y1": float(b.y1), "x2": float(b.x2), "y2": float(b.y2)},
                        "confidence": float(det.confidence),
                        "class_id": int(det.class_id),
                    }
                )

            # Players (store bbox if available)
            for p in player_keypoints or []:
                if hasattr(p, "bbox"):
                    b = p.bbox
                    record["player_keypoints"].append(
                        {
                            "bbox": {"x1": float(b.x1), "y1": float(b.y1), "x2": float(b.x2), "y2": float(b.y2)},
                        }
                    )

            f.write(json.dumps(record) + "\n")

            frame_index += 1
            if frame_index % 100 == 0:
                print(f"  Processed {frame_index} frames...")

    cap.release()
    print(f"✅ JSONL written to: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run volleyball video analysis and save annotated videos and/or JSON.")
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--mode",
        choices=["detection", "classification", "both", "analysis"],
        default="analysis",
        help="Which analysis to run.",
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("detection", "both"):
        detection_out = output_dir / "object_detection.mp4"
        print(f"Running object detection...\n  input: {video_path}\n  output: {detection_out}")
        run_object_detection(str(video_path), str(detection_out))

    if args.mode in ("classification", "both"):
        classification_out = output_dir / "video_classification.mp4"
        print(f"Running video classification...\n  input: {video_path}\n  output: {classification_out}")
        run_video_classification(str(video_path), str(classification_out))

    if args.mode == "analysis":
        jsonl_out = output_dir / "frame_results.jsonl"
        print(f"Running JSON analysis...\n  input: {video_path}\n  output: {jsonl_out}")
        run_json_analysis(str(video_path), str(jsonl_out))

    print("\n✅ Done.")
    print(f"Results saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()