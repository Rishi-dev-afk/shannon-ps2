"""CSV exports and neutral engineer-facing summary."""

import csv
import html
import json
from pathlib import Path

from .privacy import require_neutral_text
from .schema import Hotspot, RiskEvent, TrackPoint


def write_outputs(
    out_dir: str | Path,
    points: list[TrackPoint],
    events: list[RiskEvent],
    hotspots: list[Hotspot],
    *, validation: dict | None = None,
) -> None:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "events.csv", [
        "event_id", "source_id", "timestamp_s", "road_user_a", "class_a",
        "road_user_b", "class_b", "conflict_type", "indicator_type", "indicator_value_s",
        "secondary_indicator_value", "severity_band", "source_type", "x_m", "y_m", "lat", "lon",
        "segment_id", "ssm_reference_value", "evidence_ref", "min_separation_m", "quality_flags",
    ], ({
        "event_id": e.event_id, "source_id": e.source_id, "timestamp_s": e.timestamp_s,
        "road_user_a": e.road_user_a, "class_a": e.class_a,
        "road_user_b": e.road_user_b, "class_b": e.class_b,
        "conflict_type": e.conflict_type, "indicator_type": e.indicator_type,
        "indicator_value_s": e.indicator_value_s,
        "secondary_indicator_value": e.secondary_indicator_value,
        "severity_band": e.severity_band, "source_type": e.source_type,
        "x_m": e.x_m, "y_m": e.y_m, "min_separation_m": e.min_separation_m,
        "lat": e.lat, "lon": e.lon, "segment_id": e.segment_id,
        "ssm_reference_value": e.ssm_reference_value, "evidence_ref": e.evidence_ref,
        "quality_flags": ";".join(e.quality_flags),
    } for e in events))
    _write_csv(directory / "hotspots.csv", [
        "cell_x", "cell_y", "event_count", "observed_road_users", "observation_minutes",
        "rate_per_1000", "rate_ci_low", "rate_ci_high", "minimum_indicator_s",
        "min_ttc_s", "min_pet_s", "dominant_conflict_type", "dominant_road_user_pair",
        "event_types", "coverage_flags", "lat", "lon",
    ], ({
        "cell_x": h.cell_x, "cell_y": h.cell_y, "event_count": h.event_count,
        "observed_road_users": h.observed_road_users,
        "observation_minutes": h.observation_minutes,
        "rate_per_1000": h.rate_per_1000, "rate_ci_low": h.rate_ci_low,
        "rate_ci_high": h.rate_ci_high,
        "minimum_indicator_s": h.minimum_indicator_s, "event_types": ";".join(h.event_types),
        "min_ttc_s": h.min_ttc_s, "min_pet_s": h.min_pet_s,
        "dominant_conflict_type": h.dominant_conflict_type,
        "dominant_road_user_pair": h.dominant_road_user_pair,
        "coverage_flags": ";".join(h.coverage_flags),
        "lat": h.lat, "lon": h.lon,
    } for h in hotspots))
    _write_geojson(directory / "hotspots.geojson", hotspots)
    _write_summary(directory / "engineer_summary.md", points, events, hotspots, validation or {})
    _write_html(directory / "engineer_report.html", directory / "engineer_summary.md")
    _write_pdf(directory / "engineer_report.pdf", directory / "engineer_summary.md")


def _write_csv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, points, events, hotspots, validation: dict) -> None:
    sources = sorted({point.source_id for point in points})
    duration = 0.0
    if points:
        duration = max(p.timestamp_s for p in points) - min(p.timestamp_s for p in points)
    road_user_counts = {}
    for point in points:
        road_user_counts[point.class_group] = road_user_counts.get(point.class_group, 0) + 1
    lines = [
        "# Road conflict screening summary", "",
        "## Coverage", "",
        f"- Sources: {', '.join(sources) if sources else 'none'}",
        f"- Track observations: {len(points)}",
        f"- Road-user observations by group: {road_user_counts}",
        f"- Unique road users: {len({(p.source_id, p.track_id) for p in points})}",
        f"- Observed time span: {duration:.2f} seconds",
        f"- Candidate events: {len(events)}", "",
        "## Ranked hotspots", "",
    ]
    if hotspots:
        for rank, hotspot in enumerate(hotspots, start=1):
            lines.append(
                f"{rank}. Metric grid cell ({hotspot.cell_x}, {hotspot.cell_y}): "
                f"{hotspot.event_count} candidate event(s); minimum indicator "
                f"{hotspot.minimum_indicator_s:.2f} s; rate per 1,000 users {hotspot.rate_per_1000}; "
                f"main pattern {hotspot.dominant_conflict_type}; pair {hotspot.dominant_road_user_pair}; "
                f"coverage flags {', '.join(hotspot.coverage_flags) or 'none'}."
            )
    else:
        lines.append("No candidate events were observed in the supplied tracks.")
    pattern_counts = {}
    for event in events:
        pattern_counts[event.conflict_type] = pattern_counts.get(event.conflict_type, 0) + 1
    lines += ["", "## Conflict patterns", ""]
    if pattern_counts:
        lines.extend(f"- {name.replace('_', ' ').title()}: {count}" for name, count in sorted(pattern_counts.items()))
    else:
        lines.append("No candidate conflict patterns were observed.")
    lines += ["", "## Validation summary", ""]
    source_comparison = validation.get("source_comparison", {})
    lines.append(f"- Seed/reference event count: {source_comparison.get('ground_truth_event_count', 0)}")
    for source_name, comparison in source_comparison.get("comparisons", {}).items():
        lines.append(f"- {source_name}: recall {comparison.get('event_recall')}; spurious events {comparison.get('spurious_events')}")
    ssm = validation.get("ssm_comparison", {})
    lines.append(f"- SUMO SSM encounters available: {ssm.get('ssm_encounters', 0)}")
    lines.append(f"- TTC/PET agreement summary: {ssm.get('mean_absolute_error_s', {})}")
    detection = validation.get("detection_metrics", {}).get("per_class", {})
    if detection:
        lines.append("- Hand-checked detector precision/recall by class:")
        lines.extend(f"  - {name}: precision {values.get('precision')}; recall {values.get('recall')}"
                     for name, values in sorted(detection.items()))
    risk = validation.get("risk_model", {})
    if risk:
        held_out = risk.get("held_out_evaluation", {})
        aggregate = held_out.get("aggregate_held_out_metrics", {})
        lines += ["", "## Random Forest proxy-label evaluation", ""]
        lines.append(f"- Model: {risk.get('model_version')}; source label: {risk.get('label_source')}.")
        lines.append(f"- Held-out location PR-AUC: {held_out.get('held_out_pr_auc')}; "
                     f"precision {aggregate.get('precision')}; recall {aggregate.get('recall')}; "
                     f"F1 {aggregate.get('f1')}; confusion counts {aggregate.get('confusion_counts')}.")
        lines.append(f"- Interpretation: {risk.get('label_caveat')}")
        lines.append("- These proxy-label scores use a different target from TTC/PET event-matching metrics and should not be compared as if they were the same evaluation.")
        if risk.get("score_note"):
            lines.append(f"- Score export note: {risk.get('score_note')}")
        else:
            lines.append(f"- Scored pair windows exported: {risk.get('score_rows', 0)}")
        importance = held_out.get("permutation_feature_importance", [])[:8]
        if importance:
            lines.append("- Held-out permutation importance (top features):")
            lines.extend(f"  - {item.get('feature')}: {item.get('mean_importance'):.5f}"
                         for item in importance)
    seeded = validation.get("seeded_scenarios", {})
    if seeded:
        lines.append(f"- Seeded scenario recall: {seeded.get('event_recall')}; extra candidate events: {seeded.get('extra_candidate_events')}")
    lines += [
        "", "## Engineering review prompts", "",
        "Review crossing visibility, signal phasing, refuge provision, and approach-speed management where the mapped data and local context support it.",
        "",
        "", "## Method and limitations", "",
        "TTC is estimated from observed positions using a short constant-velocity "
        "prediction and buffered road-user footprints. Results are screening "
        "indicators for engineering review, not crash probabilities or legal findings.", "",
        "Coordinates are assumed to be calibrated metres in one shared road-plane "
        "coordinate system. Hotspots are metric grid cells and are not yet snapped "
        "to road segments. Data gaps, detector errors, and incomplete observation "
        "coverage can hide conflicts; ‘none observed’ does not mean a location is safe.",
        "Threshold defaults are TTC < 3 s, PET < 2 s, and DRAC > 3 m/s², following "
        "[SUMO SSM defaults](https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html). "
        "Treat these as configurable screening values, not universal cutoffs.",
        "",
    ]
    content = "\n".join(lines)
    require_neutral_text(content, context="engineer summary")
    path.write_text(content, encoding="utf-8")


def _write_geojson(path: Path, hotspots: list[Hotspot]) -> None:
    features = []
    for hotspot in hotspots:
        if hotspot.lon is None or hotspot.lat is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [hotspot.lon, hotspot.lat]},
            "properties": {
                "event_count": hotspot.event_count,
                "rate_per_1000": hotspot.rate_per_1000,
                "min_indicator_s": hotspot.minimum_indicator_s,
                "dominant_conflict_type": hotspot.dominant_conflict_type,
                "min_ttc_s": hotspot.min_ttc_s,
                "min_pet_s": hotspot.min_pet_s,
                "coverage_flags": list(hotspot.coverage_flags),
            },
        })
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _write_html(path: Path, summary_path: Path) -> None:
    content = summary_path.read_text(encoding="utf-8")
    require_neutral_text(content, context="HTML report")
    escaped = html.escape(content)
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Road conflict screening report</title>"
        "<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}"
        "pre{white-space:pre-wrap;line-height:1.5}</style><pre>" + escaped + "</pre>",
        encoding="utf-8",
    )


def _write_pdf(path: Path, summary_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return
    content = summary_path.read_text(encoding="utf-8")
    require_neutral_text(content, context="PDF report")
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    text = pdf.beginText(48, height - 48)
    text.setFont("Helvetica", 9)
    for line in content.splitlines():
        safe_line = line.encode("ascii", "replace").decode("ascii")
        for start in range(0, max(1, len(safe_line)), 100):
            text.textLine(safe_line[start:start + 100])
            if text.getY() < 48:
                pdf.drawText(text)
                pdf.showPage()
                text = pdf.beginText(48, height - 48)
                text.setFont("Helvetica", 9)
    pdf.drawText(text)
    pdf.save()
