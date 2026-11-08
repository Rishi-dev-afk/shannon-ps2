from pathlib import Path
import csv
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_zero import risk_model
from vision_zero.schema import TrackPoint


def _track(track_id, x_at, *, cls="car", end=6.0):
    rows = []
    count = int(end * 10)
    for i in range(count + 1):
        t = i / 10
        rows.append(risk_model.Observation(t, x_at(t), 0.0, cls))
    return rows


def test_features_use_only_observations_at_or_before_anchor():
    left = _track("a", lambda t: t, end=3)
    right = _track("b", lambda t: 5 - t, end=3, cls="person")
    anchor = 1.0
    base = risk_model._features(
        risk_model._history(left, anchor),
        risk_model._history(right, anchor),
    )
    left.append(risk_model.Observation(2.0, 1000, 0, "car"))
    right.append(risk_model.Observation(2.0, -1000, 0, "person"))
    after = risk_model._features(
        risk_model._history(left, anchor),
        risk_model._history(right, anchor),
    )
    assert base == after


def test_label_targets_conflict_in_following_two_seconds_and_gaps_are_skipped():
    session = "Location_A|series=x|sensor=s|record=1"
    tracks = {"a": _track("a", lambda t: t), "b": _track("b", lambda t: 5 - t, cls="person")}
    conflicts = {(session, ("a", "b")): [1.5, 4.5]}
    windows = list(risk_model._iter_windows(
        session, "Location_A", tracks, conflicts, sample_every_s=1.0, radius_m=50, training=True,
    ))
    at_one = next(w for w in windows if w.window_end_s == 1.0)
    assert at_one.label_source_flag == 1
    assert at_one.window_start_s == 0.0

    gap_track = [
        risk_model.Observation(0.0, 0, 0, "car"),
        risk_model.Observation(0.1, 0.1, 0, "car"),
        risk_model.Observation(0.2, 0.2, 0, "car"),
        risk_model.Observation(1.0, 1.0, 0, "car"),
        risk_model.Observation(1.1, 1.1, 0, "car"),
    ]
    assert risk_model._history(gap_track, 1.1) is None


def test_missing_tracks_produce_no_windows():
    session = "s"
    assert list(risk_model._iter_windows(
        session, "Location_A", {"only": _track("only", lambda t: t)},
        {}, sample_every_s=1, radius_m=50, training=True,
    )) == []


def _write_training_windows(path):
    columns = [
        "location", "session_id", "road_user_a", "road_user_b", "class_a", "class_b",
        "window_start_s", "window_end_s", "label_source_flag", *risk_model.NUMERIC_FEATURES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for location in ("Location_A", "Location_B"):
            for session_index in range(6):
                session = f"{location}-record-{session_index}"
                for row_index in range(20):
                    label = int(row_index == 0)
                    writer.writerow({
                        "location": location, "session_id": session,
                        "road_user_a": "a", "road_user_b": "b", "class_a": "car", "class_b": "person",
                        "window_start_s": row_index, "window_end_s": row_index + 1,
                        "label_source_flag": label,
                        "distance_at_window_end_m": 3 + 10 * (1 - label),
                        "minimum_distance_past_1s_m": 2 + 8 * (1 - label),
                        "distance_change_past_1s_m": -2 + label,
                        "closing_speed_mps": 4 * label,
                        "relative_speed_mps": 5 + label,
                        "speed_a_mps": 3, "speed_b_mps": 2,
                        "heading_alignment": -0.8 + 1.6 * label,
                    })


def test_grouped_training_saves_model_and_exports_scores(tmp_path):
    pytest.importorskip("sklearn")
    prepared = tmp_path / "pair_windows.csv"
    model_path = tmp_path / "rf.joblib"
    _write_training_windows(prepared)
    assert risk_model._forest(risk_model._sklearn()).class_weight == "balanced_subsample"
    result = risk_model.train_risk_model(prepared, model_path)
    evaluation = result["evaluation"]
    assert model_path.exists()
    assert model_path.with_suffix(".schema.json").exists()
    assert {f["train_location"] + "->" + f["test_location"] for f in evaluation["folds"]} == {
        "Location_A->Location_B", "Location_B->Location_A",
    }
    for fold in evaluation["folds"]:
        assert fold["fit_recordings"] + fold["validation_recordings"] == fold["train_recordings"]
        assert fold["test_recordings"] > 0
    assert evaluation["aggregate_held_out_metrics"]["confusion_counts"]["true_positive"] >= 0
    assert evaluation["permutation_feature_importance"]

    points = []
    for i in range(41):
        t = i / 10
        points.append(TrackPoint("clip", t, "a", "car", t, 0, source_type="video"))
        points.append(TrackPoint("clip", t, "b", "person", 5 - t, 0, source_type="video"))
    rows, metadata = risk_model.score_track_points(model_path, points)
    assert rows
    assert metadata["label_definition"] == risk_model.LABEL_DEFINITION
    target = tmp_path / "model_scores.csv"
    risk_model.write_model_scores(target, rows)
    with target.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == len(rows)
    assert {"score", "decision", "model_version", "label_source_metadata"} <= set(exported[0])
    assert "source_method" in exported[0]["decision"] or "below_proxy" in exported[0]["decision"]
    from vision_zero.pipeline import analyze_run
    run_dir = analyze_run(points, tmp_path, run_id="risk-enabled", source_type="video", risk_model_path=model_path)
    assert (run_dir / "model_scores.csv").exists()
    validation = __import__("json").loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["risk_model"]["held_out_evaluation"]["held_out_pr_auc"] is not None
    summary = (run_dir / "engineer_summary.md").read_text(encoding="utf-8")
    assert "source-method" in summary.lower() or "source method" in summary.lower()


def test_prepare_risk_data_reads_per_record_source_files(tmp_path):
    root = tmp_path / "dataset"
    for location in ("Location_A", "Location_B"):
        base = root / f"{location}_Trajectories_and_Conflicts"
        (base / "Trajectories").mkdir(parents=True)
        conflict_dir = base / "Conflicts" / "PET"
        conflict_dir.mkdir(parents=True)
        trajectory = base / "Trajectories" / "record_1_enhanced_trajectories.csv"
        with trajectory.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "idMeasurementSeries", "idSensor", "idRecord", "FrameTimeStamp",
                "FrameTimeStamp_MicroSec", "idVehicle", "ObjectClass", "posXFit", "posYFit",
            ])
            writer.writeheader()
            for i in range(61):
                t = i / 10
                for vehicle, cls, x in (
                    ("a", 210, t),
                    ("b", 810, 5 - t),
                ):
                    writer.writerow({
                        "idMeasurementSeries": 10, "idSensor": 20001, "idRecord": 1,
                        "FrameTimeStamp": t, "FrameTimeStamp_MicroSec": 0,
                        "idVehicle": vehicle, "ObjectClass": cls, "posXFit": x, "posYFit": 0,
                    })
        conflict = conflict_dir / "record_1_Conflicts.csv"
        with conflict.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "idMeasurementSeries", "idSensor", "idRecord", "FrameTimeStamp",
                "FrameTimeStamp_MicroSec", "idVehicle1", "idVehicle2", "Value", "LOF1",
            ])
            writer.writeheader()
            writer.writerow({
                "idMeasurementSeries": 10, "idSensor": 20001, "idRecord": 1,
                "FrameTimeStamp": 2.5, "FrameTimeStamp_MicroSec": 0,
                "idVehicle1": "a", "idVehicle2": "b", "Value": 0.3, "LOF1": 0.7,
            })
    output = tmp_path / "prepared.csv"
    schema = risk_model.prepare_risk_data(root, output)
    assert output.exists()
    assert output.with_suffix(".schema.json").exists()
    assert schema["counts"]["sessions"] == 2
    assert schema["counts"]["flagged_proxy"] > 0
    assert schema["counts"]["not_flagged_proxy"] > 0
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "label_source_flag" in row
    assert "Value" not in row and "LOF1" not in row


def test_detector_metrics_only_score_reviewed_frames_and_group_vehicle_classes(tmp_path):
    from vision_zero.detection_metrics import detection_precision_recall
    labels = tmp_path / "labels.csv"
    predictions = tmp_path / "predictions.csv"
    manifest = tmp_path / "frames.csv"
    labels.write_text(
        "frame_id,class,x1,y1,x2,y2\n1,vehicle,0,0,10,10\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "frame_id,class,x1,y1,x2,y2,confidence\n"
        "1,car,0,0,10,10,0.9\n"
        "2,truck,0,0,10,10,0.8\n"
        "3,bus,0,0,10,10,0.7\n",
        encoding="utf-8",
    )
    manifest.write_text("frame_id,time_s,image_file\n1,0.0,frame1.jpg\n2,1.0,frame2.jpg\n", encoding="utf-8")
    result = detection_precision_recall(labels, predictions, iou_threshold=0.5, evaluated_frames_csv=manifest)
    assert result["evaluated_frame_count"] == 2
    assert result["per_class"]["vehicle"]["true_positive"] == 1
    assert result["per_class"]["vehicle"]["false_positive"] == 1
    assert result["per_class"]["vehicle"]["false_negative"] == 0
