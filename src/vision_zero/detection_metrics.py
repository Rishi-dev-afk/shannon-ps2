"""Frame-level object detection precision/recall against a hand-checked CSV."""

import csv
from collections import defaultdict
from pathlib import Path


def detection_precision_recall(labels_csv: str | Path, predictions_csv: str | Path,
                              *, iou_threshold: float = 0.5,
                              evaluated_frames_csv: str | Path | None = None) -> dict:
    """Score hand-labeled frames only; optional manifest includes reviewed empty frames."""
    labels = _read_boxes(labels_csv)
    predictions = _read_boxes(predictions_csv)
    if evaluated_frames_csv:
        evaluated_frames = _read_frame_manifest(evaluated_frames_csv)
    else:
        evaluated_frames = {row["frame_id"] for row in labels}
    if not evaluated_frames:
        raise ValueError(
            "No evaluated frames were supplied. Add at least one labeled frame or pass the sampled-frame manifest."
        )
    predictions = [row for row in predictions if row["frame_id"] in evaluated_frames]
    grouped_truth: dict[str, list[dict]] = defaultdict(list)
    grouped_pred: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        if row["frame_id"] in evaluated_frames:
            grouped_truth[row["frame_id"]].append(row)
    for row in predictions:
        grouped_pred[row["frame_id"]].append(row)
    classes = sorted({r["class"] for r in labels} | {r["class"] for r in predictions})
    result = {}
    for category in classes:
        tp = fp = fn = 0
        for frame_id in evaluated_frames:
            truth = [r for r in grouped_truth[frame_id] if r["class"] == category]
            pred = [r for r in grouped_pred[frame_id] if r["class"] == category]
            candidates = sorted(
                ((_iou(p, t), pi, ti) for pi, p in enumerate(pred) for ti, t in enumerate(truth)),
                reverse=True,
            )
            used_pred, used_truth = set(), set()
            for overlap, pi, ti in candidates:
                if overlap >= iou_threshold and pi not in used_pred and ti not in used_truth:
                    used_pred.add(pi)
                    used_truth.add(ti)
            tp += len(used_pred)
            fp += len(pred) - len(used_pred)
            fn += len(truth) - len(used_truth)
        result[category] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return {
        "iou_threshold": iou_threshold,
        "evaluated_frame_count": len(evaluated_frames),
        "evaluated_frames_manifest": str(evaluated_frames_csv) if evaluated_frames_csv else None,
        "per_class": result,
    }


def _canonical_frame_id(value) -> str:
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def _class_group(value: str) -> str:
    name = value.strip().lower().replace("_", " ")
    if name in {"car", "bus", "truck", "van", "vehicle", "motor vehicle"}:
        return "vehicle"
    if name in {"person", "pedestrian", "human"}:
        return "person"
    if name in {"bike", "bicycle"}:
        return "bicycle"
    if name in {"motorcycle", "motorbike"}:
        return "motorcycle"
    return name


def _read_frame_manifest(path: str | Path) -> set[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "frame_id" not in (reader.fieldnames or []):
            raise ValueError("Frame manifest must contain a frame_id column.")
        return {_canonical_frame_id(row["frame_id"]) for row in reader if row.get("frame_id")}




def _read_boxes(path):
    required = {"frame_id", "class", "x1", "y1", "x2", "y2"}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing box CSV fields: {', '.join(sorted(missing))}")
        return [{**row, "frame_id": _canonical_frame_id(row["frame_id"]), "class": _class_group(row["class"]),
                 **{k: float(row[k]) for k in ("x1", "y1", "x2", "y2")}} for row in reader]


def _iou(a, b):
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def sample_video_frames_for_annotation(video_path: str | Path, output_dir: str | Path, *, count: int = 100) -> dict:
    """Export evenly spaced source frames and a manifest for hand annotation."""
    if count < 1:
        raise ValueError("count must be at least one")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Frame sampling requires the pipeline extra with OpenCV.") from exc
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if total < 1:
        capture.release()
        raise ValueError("The source video reports no readable frames.")
    indices = sorted(set(round(i * (total - 1) / max(1, count - 1)) for i in range(count)))
    manifest_path = output_dir / "annotation_frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "time_s", "image_file"])
        writer.writeheader()
        for frame_id in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                continue
            image_name = f"frame_{frame_id:06d}.jpg"
            if not cv2.imwrite(str(output_dir / image_name), frame):
                capture.release()
                raise OSError(f"Could not write sampled frame: {image_name}")
            writer.writerow({
                "frame_id": frame_id,
                "time_s": f"{frame_id / fps:.4f}" if fps > 0 else "",
                "image_file": image_name,
            })
    capture.release()
    template_path = output_dir / "detection_labels_template.csv"
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["frame_id", "class", "x1", "y1", "x2", "y2"])
    return {
        "source_video": str(video_path),
        "frame_count": total,
        "sampled_frames": len(indices),
        "manifest": str(manifest_path),
        "labels_template": str(template_path),
        "annotation_note": "Add one row per hand-labeled box using original pixel coordinates and the sampled frame_id.",
    }
