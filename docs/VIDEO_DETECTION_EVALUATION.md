# Video detector evaluation

The committed clip-01 output is an annotated result produced by the YOLOv4 Darknet fallback, not the original independent input and not a YOLO26 result. The original permitted source clip is external to this repository. The committed hand labels cover only frames 150 and 420, so they are a small preliminary check; the requested 100-frame YOLO26n + ByteTrack evaluation still needs that original clip and fresh annotation.

## Sample 100 source frames

Keep the source video outside Git, then run:

    python -m vision_zero.cli sample-video-frames PATH_TO_PERMITTED_ORIGINAL_CLIP inputs/private/clip-annotation --count 100

This writes JPEG frames, annotation_frame_manifest.csv, and detection_labels_template.csv. Add one CSV row per manually labeled box with its frame_id and original pixel coordinates. The manifest is passed separately so reviewed frames with zero objects count as reviewed too.

## Score predictions at IoU 0.5

Run the selected YOLO model on that same source clip and retain its frame indices:

    python -m vision_zero.cli detect PATH_TO_PERMITTED_ORIGINAL_CLIP --weights PATH_TO_YOLO26N_WEIGHTS --source-id clip-01 --run-id clip-01-yolo26n --detection-labels inputs/private/clip-annotation/detection_labels_template.csv --detection-frames-csv inputs/private/clip-annotation/annotation_frame_manifest.csv

The run bundle will include per-class precision and recall in validation.json and the engineer report. Predictions outside the manifest are excluded. The metric groups car, bus, truck, and van as vehicle, matching the project's current annotation classes. Keep the raw clip and weights out of Git.

For the existing two-frame hand-check, scoring all full-frame predictions only on frames 150 and 420 at IoU 0.5 gives vehicle precision 0.724 and recall 0.618; person precision 1.000 and recall 0.200. The earlier VIDEO_RESULTS.md near/mid-field-only score is a different restricted subset. Neither result is a 100-frame evaluation.
