"""Offline run orchestration and versioned result bundle creation."""

import json
from dataclasses import asdict, fields
from pathlib import Path

from .hotspots import aggregate_hotspots
from .indicators import detect_ttc_events
from .pet import detect_pet_events
from .report import write_outputs
from .schema import TrackPoint
from .validation import compare_sources, compare_with_ssm, threshold_sensitivity, validate_seeded_events


def analyze_run(
    points: list[TrackPoint],
    output_root: str | Path,
    *,
    run_id: str,
    source_type: str,
    max_ttc_s: float = 3.0,
    max_pet_s: float = 2.0,
    cell_size_m: float = 25.0,
    zones: dict | None = None,
    ssm_rows: list[dict] | None = None,
    detection_metrics: dict | None = None,
    expected_events_csv: str | Path | None = None,
    risk_model_path: str | Path | None = None,
) -> Path:
    run_dir = Path(output_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_points = [p for p in points if p.coordinate_system != "pixel"]
    event_sources = sorted({p.source_type for p in metric_points})
    ttc = []
    pet = []
    for event_source in event_sources:
        selected = [p for p in metric_points if p.source_type == event_source]
        ttc.extend(detect_ttc_events(selected, max_ttc_s=max_ttc_s))
        if zones:
            pet.extend(detect_pet_events(selected, zones, max_pet_s=max_pet_s))
    events = sorted(ttc + pet, key=lambda e: (e.timestamp_s, e.indicator_type))
    hotspots = aggregate_hotspots(events, points, cell_size_m=cell_size_m)
    risk_scores = []
    risk_model_summary = None
    if risk_model_path:
        from .risk_model import score_track_points, write_model_scores
        risk_scores, risk_model_summary = score_track_points(risk_model_path, points)
    validation = {
        "source_comparison": compare_sources(points, max_ttc_s=max_ttc_s),
        "threshold_sensitivity": threshold_sensitivity(metric_points, base_ttc_s=max_ttc_s),
        "ssm_comparison": compare_with_ssm(events, ssm_rows or []),
        "ssm_reference_rows": len(ssm_rows or []),
        "detection_metrics": detection_metrics or {},
        "seeded_scenarios": (validate_seeded_events(events, expected_events_csv)
                             if expected_events_csv else {}),
        "quality_banner": _quality_banner(points),
        "risk_model": risk_model_summary or {},
    }
    write_outputs(run_dir, points, events, hotspots, validation=validation)
    if risk_model_path:
        write_model_scores(run_dir / "model_scores.csv", risk_scores)
    _write_tracks(run_dir / "tracks.csv", points)
    _write_parquet_bundle(run_dir, points, events, hotspots)
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    duration = max((p.timestamp_s for p in points), default=0) - min((p.timestamp_s for p in points), default=0)
    config = {
        "run_id": run_id,
        "source_type": source_type,
        "observation_duration_s": max(0.0, duration),
        "track_observations": len(points),
        "track_count": len({(p.source_id, p.track_id) for p in points}),
        "max_ttc_s": max_ttc_s,
        "max_pet_s": max_pet_s,
        "cell_size_m": cell_size_m,
        "method_note": "Candidate screening from precomputed tracks; metric outputs require shared road-plane coordinates.",
        "thresholds": {"ttc_s": max_ttc_s, "pet_s": max_pet_s, "drac_mps2": 3.0},
        "threshold_sources": [
            {"name": "SUMO SSM default thresholds", "url": "https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html",
             "values": ["TTC < 3 s", "PET < 2 s", "DRAC > 3 m/s²"]}
        ],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir


def _write_tracks(path: Path, points: list[TrackPoint]) -> None:
    import csv
    fields = ["source_id", "source_type", "timestamp_s", "track_id", "class", "class_group",
              "x", "y", "coordinate_system", "confidence", "quality_flags"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow({
                "source_id": p.source_id, "source_type": p.source_type,
                "timestamp_s": p.timestamp_s, "track_id": p.track_id,
                "class": p.road_user_class, "class_group": p.class_group,
                "x": p.x_m, "y": p.y_m, "coordinate_system": p.coordinate_system,
                "confidence": p.confidence, "quality_flags": ";".join(p.quality_flags),
            })


def _quality_banner(points: list[TrackPoint]) -> str:
    if not points:
        return "No track observations are available."
    if any(p.coordinate_system == "pixel" for p in points):
        return "Pixel-only tracks: metric indicators and geographic hotspots are unavailable."
    if any(p.quality_flags for p in points):
        return "Some tracks have quality flags; review them before interpreting event rankings."
    return "Metric tracks available; results remain screening indicators and depend on coverage."


def _write_parquet_bundle(run_dir: Path, points, events, hotspots) -> None:
    import pandas as pd
    pd.DataFrame([asdict(point) for point in points], columns=[f.name for f in fields(TrackPoint)]).to_parquet(
        run_dir / "tracks.parquet", index=False)
    from .schema import RiskEvent, Hotspot
    pd.DataFrame([asdict(event) for event in events], columns=[f.name for f in fields(RiskEvent)]).to_parquet(
        run_dir / "events.parquet", index=False)
    pd.DataFrame([asdict(hotspot) for hotspot in hotspots], columns=[f.name for f in fields(Hotspot)]).to_parquet(
        run_dir / "hotspots.parquet", index=False)
