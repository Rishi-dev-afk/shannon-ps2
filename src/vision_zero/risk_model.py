"""Knowledge PET3D proxy-label preparation, grouped Random Forest evaluation, and scoring."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from bisect import bisect_left, bisect_right
from typing import Iterator

MODEL_VERSION = "knowledge-pet3d-rf-v1"
LABEL_SOURCE = "Knowledge PET3D published conflict output (algorithm-generated proxy)"
LABEL_DEFINITION = "source_method_flagged_conflict_within_next_2_seconds"
HISTORY_SECONDS, FORECAST_SECONDS = 1.0, 2.0
DEFAULT_SAMPLE_EVERY_S, DEFAULT_CANDIDATE_RADIUS_M = 1.0, 50.0
MAX_OBSERVATION_GAP_S = 0.5
NUMERIC_FEATURES = (
    "distance_at_window_end_m", "minimum_distance_past_1s_m",
    "distance_change_past_1s_m", "closing_speed_mps", "relative_speed_mps",
    "speed_a_mps", "speed_b_mps", "heading_alignment",
)
CATEGORICAL_FEATURES = ("class_a", "class_b")
OUTPUT_COLUMNS = (
    "location", "session_id", "road_user_a", "road_user_b", "class_a", "class_b",
    "window_start_s", "window_end_s", "label_source_flag", *NUMERIC_FEATURES,
)


@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    x_m: float
    y_m: float
    road_user_class: str


@dataclass(frozen=True)
class PairWindow:
    location: str
    session_id: str
    road_user_a: str
    road_user_b: str
    class_a: str
    class_b: str
    window_start_s: float
    window_end_s: float
    label_source_flag: int | None
    features: dict[str, float]


def _sklearn():
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:
        raise RuntimeError('Risk-model commands require the pipeline extra: python -m pip install -e ".[pipeline]".') from exc
    return locals()


def _pick(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _timestamp(row: dict) -> float | None:
    base = _pick(row, "FrameTimeStamp", "TimeStamp")
    micros = _number(_pick(row, "FrameTimeStamp_MicroSec", "TimeStamp_MicroSec", default="0"), 0.0) or 0.0
    if not base:
        return None
    numeric = _number(base)
    if numeric is not None:
        fractional_in_value = not numeric.is_integer()
        return numeric if fractional_in_value else numeric + micros / 1_000_000.0
    try:
        value = datetime.fromisoformat(base.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp() + micros / 1_000_000.0
    except ValueError:
        return None


def _session_id(row: dict, location: str, fallback: str) -> str:
    series = _pick(row, "idMeasurementSeries", default=location)
    sensor = _pick(row, "idSensor", default="sensor")
    record = _pick(row, "idRecord", default=fallback)
    return f"{location}|series={series}|sensor={sensor}|record={record}"


def _infer_location(path: Path) -> str | None:
    joined = "/".join(p.lower() for p in path.parts)
    if "location_a" in joined or "location-a" in joined:
        return "Location_A"
    if "location_b" in joined or "location-b" in joined:
        return "Location_B"
    return None


def _header(path: Path) -> set[str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return set(next(csv.reader(f), []))
    except (OSError, UnicodeError):
        return set()


def _dataset_files(root: Path):
    trajectories, conflicts = [], []
    for path in sorted(root.rglob("*.csv")):
        if _infer_location(path) is None or any(x.lower() in {"video", "videos"} for x in path.parts):
            continue
        fields = _header(path)
        location = _infer_location(path)
        if {"idVehicle", "posXFit", "posYFit"} <= fields:
            trajectories.append((location, path))
        elif {"idVehicle1", "idVehicle2"} <= fields or "conflict" in path.stem.lower():
            conflicts.append((location, path))
    if not trajectories:
        raise ValueError(f"No PET3D trajectory CSVs found below {root}; expected idVehicle/posXFit/posYFit.")
    if not conflicts:
        raise ValueError(f"No PET3D published conflict CSVs found below {root}.")
    locations = {loc for loc, _ in trajectories}
    if not {"Location_A", "Location_B"} <= locations:
        raise ValueError(f"Both Location_A and Location_B trajectories are required; found {sorted(locations)}.")
    return trajectories, conflicts


def _load_conflicts(files):
    result = defaultdict(list)
    for location, path in files:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                a = _pick(row, "idVehicle1", "idVehicle_1", "Vehicle1", "vehicle_a")
                b = _pick(row, "idVehicle2", "idVehicle_2", "Vehicle2", "vehicle_b")
                stamp = _timestamp(row)
                if a and b and stamp is not None:
                    session = _session_id(row, location, path.stem)
                    result[(session, tuple(sorted((a, b))))].append(stamp)
    for values in result.values():
        values.sort()
    return result


def _iter_trajectory_sessions(files):
    """Read one per-record CSV at a time to avoid loading the full archive into RAM."""
    seen_sessions = set()
    for location, path in files:
        file_sessions = defaultdict(lambda: defaultdict(list))
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                track_id = _pick(row, "idVehicle", "track_id")
                stamp = _timestamp(row)
                x, y = _number(_pick(row, "posXFit", "x_m")), _number(_pick(row, "posYFit", "y_m"))
                if track_id and stamp is not None and x is not None and y is not None:
                    session = _session_id(row, location, path.stem)
                    cls = _pick(row, "ObjectClass", "class", default="unknown").lower()
                    file_sessions[session][track_id].append(Observation(stamp, x, y, cls))
        for session, tracks in file_sessions.items():
            if session in seen_sessions:
                raise ValueError(
                    f"Session {session} appears in multiple trajectory files. "
                    "Keep one source CSV per recording or merge those files before preparation."
                )
            seen_sessions.add(session)
            for track_id, values in tracks.items():
                unique = {p.timestamp_s: p for p in sorted(values, key=lambda p: p.timestamp_s)}
                tracks[track_id] = list(unique.values())
            yield session, tracks


def _relative_tracks(tracks):
    offset = min(p.timestamp_s for values in tracks.values() for p in values)
    return {
        tid: [Observation(p.timestamp_s - offset, p.x_m, p.y_m, p.road_user_class) for p in values]
        for tid, values in tracks.items() if len(values) >= 2
    }, offset


def _bbox_gap(a, b):
    dx = max(0.0, min(p.x_m for p in b) - max(p.x_m for p in a), min(p.x_m for p in a) - max(p.x_m for p in b))
    dy = max(0.0, min(p.y_m for p in b) - max(p.y_m for p in a), min(p.y_m for p in a) - max(p.y_m for p in b))
    return math.hypot(dx, dy)


def _history(track, anchor, duration=HISTORY_SECONDS):
    times = [p.timestamp_s for p in track]
    selected = track[bisect_left(times, anchor - duration):bisect_right(times, anchor)]
    if len(selected) < 2 or anchor - selected[-1].timestamp_s > MAX_OBSERVATION_GAP_S:
        return None
    if selected[0].timestamp_s - (anchor - duration) > MAX_OBSERVATION_GAP_S:
        return None
    if any(b.timestamp_s - a.timestamp_s > MAX_OBSERVATION_GAP_S for a, b in zip(selected, selected[1:])):
        return None
    return selected


def _speed(history):
    a, b = history[-2], history[-1]
    dt = b.timestamp_s - a.timestamp_s
    if dt <= 0:
        return 0.0, 0.0, 0.0
    vx, vy = (b.x_m - a.x_m) / dt, (b.y_m - a.y_m) / dt
    return vx, vy, math.hypot(vx, vy)


def _features(a, b):
    a0, a1, b0, b1 = a[0], a[-1], b[0], b[-1]
    rx, ry = b1.x_m - a1.x_m, b1.y_m - a1.y_m
    distance = math.hypot(rx, ry)
    start_distance = math.hypot(b0.x_m - a0.x_m, b0.y_m - a0.y_m)
    avx, avy, speed_a = _speed(a)
    bvx, bvy, speed_b = _speed(b)
    rvx, rvy = bvx - avx, bvy - avy
    closing = -(rx * rvx + ry * rvy) / max(distance, 1e-6)
    alignment = ((avx * bvx + avy * bvy) / max(speed_a * speed_b, 1e-6)
                 if speed_a > 1e-6 and speed_b > 1e-6 else 0.0)
    times_a, times_b = [p.timestamp_s for p in a], [p.timestamp_s for p in b]
    distances = []
    for stamp in sorted(set(times_a) | set(times_b)):
        ia, ib = bisect_right(times_a, stamp) - 1, bisect_right(times_b, stamp) - 1
        if ia >= 0 and ib >= 0:
            pa, pb = a[ia], b[ib]
            if stamp - pa.timestamp_s <= MAX_OBSERVATION_GAP_S and stamp - pb.timestamp_s <= MAX_OBSERVATION_GAP_S:
                distances.append(math.hypot(pb.x_m - pa.x_m, pb.y_m - pa.y_m))
    return {
        "distance_at_window_end_m": distance,
        "minimum_distance_past_1s_m": min(distances, default=distance),
        "distance_change_past_1s_m": distance - start_distance,
        "closing_speed_mps": closing,
        "relative_speed_mps": math.hypot(rvx, rvy),
        "speed_a_mps": speed_a, "speed_b_mps": speed_b,
        "heading_alignment": max(-1.0, min(1.0, alignment)),
    }


def _covered(track, target):
    times = [p.timestamp_s for p in track]
    i = bisect_left(times, target - MAX_OBSERVATION_GAP_S)
    return i < len(times) and times[i] <= target + MAX_OBSERVATION_GAP_S


def _iter_windows(session, location, absolute_tracks, conflicts, *, sample_every_s, radius_m, training):
    if not absolute_tracks:
        return
    tracks, offset = _relative_tracks(absolute_tracks)
    local_conflicts = {pair: [t - offset for t in times] for (sid, pair), times in conflicts.items() if sid == session}
    for (id1, track1), (id2, track2) in combinations(sorted(tracks.items()), 2):
        pair = tuple(sorted((id1, id2)))
        event_times = local_conflicts.get(pair, [])
        if _bbox_gap(track1, track2) > radius_m and not event_times:
            continue
        start = max(track1[0].timestamp_s, track2[0].timestamp_s) + HISTORY_SECONDS
        end = min(track1[-1].timestamp_s, track2[-1].timestamp_s) - (FORECAST_SECONDS if training else 0.0)
        if end < start:
            continue
        anchor = math.ceil((start - 1e-9) / sample_every_s) * sample_every_s
        while anchor <= end + 1e-9:
            ha, hb = _history(track1, anchor), _history(track2, anchor)
            if ha and hb:
                feats = _features(ha, hb)
                label = any(anchor < t <= anchor + FORECAST_SECONDS for t in event_times)
                if feats["minimum_distance_past_1s_m"] <= radius_m or label:
                    if training and not (_covered(track1, anchor + FORECAST_SECONDS) and _covered(track2, anchor + FORECAST_SECONDS)):
                        anchor += sample_every_s
                        continue
                    yield PairWindow(
                        location, session, id1, id2, ha[-1].road_user_class, hb[-1].road_user_class,
                        anchor - HISTORY_SECONDS, anchor, int(label) if training else None, feats,
                    )
            anchor += sample_every_s


def prepare_risk_data(dataset_root, output_csv, *, sample_every_s=DEFAULT_SAMPLE_EVERY_S,
                      candidate_radius_m=DEFAULT_CANDIDATE_RADIUS_M):
    if sample_every_s <= 0 or candidate_radius_m <= 0:
        raise ValueError("sample interval and candidate radius must be positive")
    trajectory_files, conflict_files = _dataset_files(Path(dataset_root))
    conflicts = _load_conflicts(conflict_files)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"windows": 0, "flagged_proxy": 0, "not_flagged_proxy": 0, "sessions": 0}
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for session, tracks in _iter_trajectory_sessions(trajectory_files):
            counts["sessions"] += 1
            location = "Location_A" if "Location_A|" in session else "Location_B"
            for w in _iter_windows(session, location, tracks, conflicts, sample_every_s=sample_every_s,
                                   radius_m=candidate_radius_m, training=True):
                writer.writerow({
                    "location": w.location, "session_id": w.session_id,
                    "road_user_a": w.road_user_a, "road_user_b": w.road_user_b,
                    "class_a": w.class_a, "class_b": w.class_b,
                    "window_start_s": f"{w.window_start_s:.6f}", "window_end_s": f"{w.window_end_s:.6f}",
                    "label_source_flag": w.label_source_flag,
                    **{k: f"{w.features[k]:.8g}" for k in NUMERIC_FEATURES},
                })
                counts["windows"] += 1
                counts["flagged_proxy" if w.label_source_flag else "not_flagged_proxy"] += 1
    if not counts["windows"]:
        raise ValueError("No complete pair windows found; check timestamps, IDs, positions, and recording coverage.")
    schema = {
        "model_version": MODEL_VERSION, "label_source": "Knowledge PET3D published conflict output",
        "label_definition": LABEL_DEFINITION,
        "label_caveat": "Not flagged by the source method is not a confirmed safe example.",
        "history_seconds": HISTORY_SECONDS, "forecast_seconds": FORECAST_SECONDS,
        "sample_every_seconds": sample_every_s, "candidate_radius_m": candidate_radius_m,
        "numeric_features": list(NUMERIC_FEATURES), "categorical_features": list(CATEGORICAL_FEATURES),
        "excluded_from_features": ["PET", "LOF1", "LOF2", "Value", "Auto_Rule_Flag", "Manual_Rule_Flag",
                                   "Manual_Conflict_Check", "conflict scores", "vehicle/session/location IDs"],
        "counts": counts,
    }
    output.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def _feature_dict(row):
    result = {}
    for name in NUMERIC_FEATURES:
        value = _number(row.get(name))
        if value is None:
            raise ValueError(f"Invalid prepared feature: {name}")
        result[name] = value
    for name in CATEGORICAL_FEATURES:
        result[f"{name}={(_pick(row, name, default='unknown').lower())}"] = 1.0
    return result


def _load_prepared(path):
    rows, labels, groups, locations = [], [], [], []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"location", "session_id", "label_source_flag", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prepared CSV missing fields: {', '.join(sorted(missing))}")
        for row in reader:
            label = _number(row.get("label_source_flag"))
            if label not in (0, 1):
                raise ValueError("label_source_flag must contain only 0 or 1")
            rows.append(_feature_dict(row))
            labels.append(int(label))
            groups.append(row["session_id"])
            locations.append(row["location"])
    if not rows:
        raise ValueError("Prepared CSV contains no rows")
    if len(set(locations)) < 2:
        raise ValueError("Training requires two locations for location-held-out evaluation")
    if len(set(labels)) < 2:
        raise ValueError("Training requires both source-flagged and not-flagged windows")
    return rows, labels, groups, locations


def _positive_probability(model, matrix):
    probabilities = model.predict_proba(matrix)
    classes = list(model.classes_)
    return [float(r[classes.index(1)]) if 1 in classes else 0.0 for r in probabilities]


def _threshold(labels, scores, tools):
    if len(set(labels)) < 2:
        return 0.5, "validation subset has one class; default threshold 0.5"
    best = None
    for value in sorted({0.0, 0.5, 1.0, *scores}):
        pred = [int(score >= value) for score in scores]
        candidate = (float(tools["f1_score"](labels, pred, zero_division=0)),
                     float(tools["precision_score"](labels, pred, zero_division=0)), float(value))
        if best is None or candidate > best:
            best = candidate
    return best[2], "threshold maximized validation F1; ties prefer precision, then higher threshold"


def _metrics(labels, scores, threshold, tools):
    pred = [int(score >= threshold) for score in scores]
    matrix = tools["confusion_matrix"](labels, pred, labels=[0, 1]).tolist()
    return {
        "pr_auc_average_precision": float(tools["average_precision_score"](labels, scores)) if len(set(labels)) > 1 else None,
        "precision": float(tools["precision_score"](labels, pred, zero_division=0)) if labels else None,
        "recall": float(tools["recall_score"](labels, pred, zero_division=0)) if labels else None,
        "f1": float(tools["f1_score"](labels, pred, zero_division=0)) if labels else None,
        "confusion_counts": {"true_negative": int(matrix[0][0]), "false_positive": int(matrix[0][1]),
                             "false_negative": int(matrix[1][0]), "true_positive": int(matrix[1][1])},
        "sample_count": len(labels), "positive_proxy_count": int(sum(labels)), "threshold": float(threshold),
    }


def _forest(tools):
    return tools["RandomForestClassifier"](
        n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )


def train_risk_model(prepared_csv, model_out, *, metrics_out=None, random_state=42):
    tools = _sklearn()
    rows, labels, groups, locations = _load_prepared(prepared_csv)
    vectorizer_type = tools["DictVectorizer"]
    vec_cv = vectorizer_type(sparse=True)
    x_all = vec_cv.fit_transform(rows)
    feature_names = list(vec_cv.get_feature_names_out())
    location_names = sorted(set(locations))
    folds, oof_labels, oof_scores, oof_pred = [], [], [], []
    importances = defaultdict(list)
    for test_location in location_names:
        train_idx = [i for i, loc in enumerate(locations) if loc != test_location]
        test_idx = [i for i, loc in enumerate(locations) if loc == test_location]
        train_groups = sorted({groups[i] for i in train_idx})
        fit_idx, val_idx = train_idx, []
        val_note = "fewer than two training recordings; default threshold 0.5"
        if len(train_groups) >= 2:
            splitter = tools["GroupShuffleSplit"](n_splits=1, test_size=0.2, random_state=random_state)
            fit_local, val_local = next(splitter.split(
                [0] * len(train_idx), [labels[i] for i in train_idx], [groups[i] for i in train_idx]
            ))
            fit_idx = [train_idx[int(i)] for i in fit_local]
            val_idx = [train_idx[int(i)] for i in val_local]
        model = _forest(tools)
        model.fit(x_all[fit_idx], [labels[i] for i in fit_idx])
        threshold, val_note = 0.5, val_note
        if val_idx:
            threshold, val_note = _threshold([labels[i] for i in val_idx], _positive_probability(model, x_all[val_idx]), tools)
        scores = _positive_probability(model, x_all[test_idx])
        test_labels = [labels[i] for i in test_idx]
        metrics = _metrics(test_labels, scores, threshold, tools)
        oof_labels.extend(test_labels)
        oof_scores.extend(scores)
        oof_pred.extend(int(s >= threshold) for s in scores)
        sample_idx = test_idx
        if len(sample_idx) > 5000:
            rng = tools["np"].random.default_rng(random_state)
            sample_idx = sorted(rng.choice(sample_idx, 5000, replace=False).tolist())
        # scikit-learn permutation_importance expects dense input with current versions.
        importance_matrix = x_all[sample_idx]
        if hasattr(importance_matrix, "toarray"):
            importance_matrix = importance_matrix.toarray()
        importance_scoring = "average_precision" if len(set(test_labels)) > 1 else "f1"
        perm = tools["permutation_importance"](
            model, importance_matrix, [labels[i] for i in sample_idx],
            n_repeats=5, random_state=random_state, scoring=importance_scoring, n_jobs=1,
        )
        for name, value in zip(feature_names, perm.importances_mean):
            importances[name].append(float(value))
        folds.append({
            "train_location": next(loc for loc in location_names if loc != test_location),
            "test_location": test_location,
            "train_recordings": len(set(groups[i] for i in train_idx)),
            "fit_recordings": len(set(groups[i] for i in fit_idx)),
            "validation_recordings": len(set(groups[i] for i in val_idx)),
            "test_recordings": len(set(groups[i] for i in test_idx)),
            "threshold": threshold, "threshold_selection": val_note, **metrics,
        })
    if len(folds) != 2:
        raise ValueError("Both Location_A→Location_B and Location_B→Location_A folds are required")
    aggregate = _metrics(oof_labels, oof_scores, 0.5, tools)
    # Recompute aggregate thresholded metrics from the fold-specific validation thresholds.
    matrix = tools["confusion_matrix"](oof_labels, oof_pred, labels=[0, 1]).tolist()
    aggregate.update({
        "precision": float(tools["precision_score"](oof_labels, oof_pred, zero_division=0)),
        "recall": float(tools["recall_score"](oof_labels, oof_pred, zero_division=0)),
        "f1": float(tools["f1_score"](oof_labels, oof_pred, zero_division=0)),
        "confusion_counts": {"true_negative": int(matrix[0][0]), "false_positive": int(matrix[0][1]),
                             "false_negative": int(matrix[1][0]), "true_positive": int(matrix[1][1])},
        "threshold": "fold-specific validation thresholds",
    })
    importance_rows = [
        {"feature": name, "mean_importance": statistics.fmean(values), "fold_count": len(values)}
        for name, values in importances.items()
    ]
    importance_rows.sort(key=lambda x: (-x["mean_importance"], x["feature"]))
    thresholds = [fold["threshold"] for fold in folds]
    evaluation = {
        "model_version": MODEL_VERSION, "label_source": "Knowledge PET3D published conflict output",
        "label_definition": LABEL_DEFINITION,
        "label_caveat": "Not flagged by the source method is not a confirmed safe example; this classifier imitates source-method flags and does not predict crashes.",
        "split_strategy": "two-way leave-one-location-out; threshold validation held out by recording within the training location",
        "folds": folds, "held_out_pr_auc": aggregate["pr_auc_average_precision"],
        "aggregate_held_out_metrics": aggregate, "permutation_feature_importance": importance_rows[:30],
        "final_model_threshold": float(statistics.median(thresholds)),
        "final_threshold_source": "median of within-training-location validation thresholds",
        "source_data_rows": len(rows), "recordings": len(set(groups)), "locations": location_names,
    }
    final_vec = vectorizer_type(sparse=True)
    final_x = final_vec.fit_transform(rows)
    final_model = _forest(tools)
    final_model.fit(final_x, labels)
    artifact = {
        "model": final_model, "vectorizer": final_vec,
        "metadata": {
            "model_version": MODEL_VERSION, "label_source": "Knowledge PET3D published conflict output",
            "label_definition": LABEL_DEFINITION, "label_caveat": evaluation["label_caveat"],
            "history_seconds": HISTORY_SECONDS, "forecast_seconds": FORECAST_SECONDS,
            "numeric_features": list(NUMERIC_FEATURES), "categorical_features": list(CATEGORICAL_FEATURES),
            "threshold": evaluation["final_model_threshold"], "evaluation": evaluation,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(), "training_rows": len(rows),
            "training_locations": location_names,
        },
    }
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tools["joblib"].dump(artifact, model_path, compress=3)
    schema = {
        "model_version": MODEL_VERSION, "label_source": artifact["metadata"]["label_source"],
        "label_definition": LABEL_DEFINITION, "history_seconds": HISTORY_SECONDS,
        "forecast_seconds": FORECAST_SECONDS, "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "expanded_model_features": list(final_vec.get_feature_names_out()),
        "excluded_from_features": ["PET", "LOF", "source conflict scores/flags", "vehicle/session IDs", "location"],
        "evaluation": evaluation,
    }
    model_path.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    result_path = Path(metrics_out) if metrics_out else model_path.with_suffix(".evaluation.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metrics_path": str(result_path), "evaluation": evaluation}


def score_track_points(model_path, points):
    tools = _sklearn()
    artifact = tools["joblib"].load(model_path)
    sessions = defaultdict(lambda: defaultdict(list))
    for p in points:
        if str(getattr(p, "coordinate_system", "local_m")).lower() == "pixel":
            continue
        session = f"{p.source_type}|{p.source_id}"
        sessions[session][str(p.track_id)].append(Observation(float(p.timestamp_s), float(p.x_m), float(p.y_m), str(p.road_user_class).lower()))
    windows = []
    for session, tracks in sessions.items():
        windows.extend(_iter_windows(session, "analysis_input", tracks, {}, sample_every_s=1.0,
                                     radius_m=DEFAULT_CANDIDATE_RADIUS_M, training=False))
    feature_rows = [
        {**w.features, f"class_a={w.class_a}": 1.0, f"class_b={w.class_b}": 1.0}
        for w in windows
    ]
    scores = _positive_probability(artifact["model"], artifact["vectorizer"].transform(feature_rows)) if feature_rows else []
    threshold = float(artifact["metadata"].get("threshold", 0.5))
    rows = []
    for w, score in zip(windows, scores):
        rows.append({
            "source_id": w.session_id.split("|", 1)[-1], "road_user_a": w.road_user_a, "road_user_b": w.road_user_b,
            "class_a": w.class_a, "class_b": w.class_b,
            "window_start_s": f"{w.window_start_s:.6f}", "window_end_s": f"{w.window_end_s:.6f}",
            "score": f"{score:.8f}",
            "decision": "source_method_flag_proxy" if score >= threshold else "below_proxy_threshold",
            "model_version": artifact["metadata"].get("model_version", "unknown"),
            "label_source_metadata": artifact["metadata"].get("label_source", ""),
            "label_definition": artifact["metadata"].get("label_definition", ""),
        })
    metadata = artifact["metadata"]
    return rows, {
        "model_version": metadata.get("model_version"), "label_source": metadata.get("label_source"),
        "label_definition": metadata.get("label_definition"), "label_caveat": metadata.get("label_caveat"),
        "threshold": threshold, "score_rows": len(rows),
        "score_note": ("No metric-coordinate pair windows were available; pixel-only tracks are not scored."
                       if not rows else None),
        "held_out_evaluation": metadata.get("evaluation", {}),
    }


def write_model_scores(path, rows):
    columns = [
        "source_id", "road_user_a", "road_user_b", "class_a", "class_b", "window_start_s",
        "window_end_s", "score", "decision", "model_version", "label_source_metadata", "label_definition",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
