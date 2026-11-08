"""Command-line entry point."""

import argparse
import json
import uuid
from pathlib import Path

from .pipeline import analyze_run
from .ingest import load_tracks
from .scenario import generate_seeded_routes
from .detection_metrics import detection_precision_recall, sample_video_frames_for_annotation
from .sumo import build_osm_network, read_fcd, read_ssm_values, run_sumo, write_sumocfg
from .vision import detect_and_track, render_track_frames
from .calibration import fit_homography
from .risk_model import prepare_risk_data, train_risk_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="vision-zero")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="analyze timestamped road-user tracks")
    analyze.add_argument("input_csv")
    analyze.add_argument("--out", default="outputs")
    analyze.add_argument("--run-id", default=None)
    analyze.add_argument("--source-type", choices=["sumo_gt", "sumo_detected", "video"], default="sumo_gt")
    analyze.add_argument("--cell-size-m", type=float, default=25.0)
    analyze.add_argument("--max-ttc-s", type=float, default=3.0)
    analyze.add_argument("--max-pet-s", type=float, default=2.0)
    analyze.add_argument("--zones-json", default=None)
    analyze.add_argument("--detection-labels", default=None)
    analyze.add_argument("--detection-predictions", default=None)
    analyze.add_argument("--detection-frames-csv", default=None, help="manifest of all reviewed frames, including empty frames")
    analyze.add_argument("--expected-events-csv", default=None)
    analyze.add_argument("--risk-model", default=None, help="optional trained Random Forest artifact for pair-window scoring")

    prepare = commands.add_parser("prepare-risk-data", help="prepare Knowledge PET3D pair windows with source-method proxy labels")
    prepare.add_argument("dataset_root")
    prepare.add_argument("output_csv")
    prepare.add_argument("--sample-every-s", type=float, default=1.0)
    prepare.add_argument("--candidate-radius-m", type=float, default=50.0)

    train = commands.add_parser("train-risk-model", help="train and location-held-out evaluate the Random Forest")
    train.add_argument("pair_windows_csv")
    train.add_argument("--model-out", default="outputs/models/knowledge-pet3d-rf.joblib")
    train.add_argument("--metrics-out", default=None)

    network = commands.add_parser("build-network", help="convert a local OSM extract to SUMO net.xml")
    network.add_argument("osm_file")
    network.add_argument("output_net")

    config = commands.add_parser("write-config", help="write a SUMO config for a network and route file")
    config.add_argument("network")
    config.add_argument("routes")
    config.add_argument("output_cfg")
    config.add_argument("--duration", type=float, default=180.0)

    routes = commands.add_parser("seed-routes", help="make repeatable routes for an OSM-derived network")
    routes.add_argument("paths_json", help="JSON with vehicle_paths and pedestrian_paths edge lists")
    routes.add_argument("output_routes")
    routes.add_argument("--seed", type=int, default=42)
    routes.add_argument("--demand", type=int, default=18)
    routes.add_argument("--conflict-mode", choices=["crossing", "turning", "bike_cut_across", "control"], default="crossing")

    simulate = commands.add_parser("simulate", help="run an existing SUMO config and analyze FCD tracks")
    simulate.add_argument("sumocfg")
    simulate.add_argument("--out", default="outputs")
    simulate.add_argument("--run-id", default=None)
    simulate.add_argument("--network", default=None)
    simulate.add_argument("--render", action="store_true")
    simulate.add_argument("--weights", default=None, help="optional YOLO weights for rendered frames")
    simulate.add_argument("--detection-labels", default=None)
    simulate.add_argument("--detection-frames-csv", default=None)

    detect = commands.add_parser("detect", help="run YOLO tracking on a video")
    detect.add_argument("video")
    detect.add_argument("--weights", required=True)
    detect.add_argument("--source-id", default="recorded_clip")
    detect.add_argument("--out", default="outputs")
    detect.add_argument("--run-id", default=None)
    detect.add_argument("--homography-json", default=None)
    detect.add_argument("--detection-labels", default=None)
    detect.add_argument("--detection-frames-csv", default=None)

    metrics = commands.add_parser("detection-metrics", help="score a hand-checked detection sample")
    metrics.add_argument("labels_csv")
    metrics.add_argument("predictions_csv")
    metrics.add_argument("--iou", type=float, default=0.5)
    metrics.add_argument("--frames-csv", default=None, help="sampled-frame manifest, including reviewed empty frames")

    sample_frames = commands.add_parser("sample-video-frames", help="sample a source clip for hand annotation")
    sample_frames.add_argument("video")
    sample_frames.add_argument("output_dir")
    sample_frames.add_argument("--count", type=int, default=100)

    calibrate = commands.add_parser("calibrate", help="fit a road-plane homography from landmarks JSON")
    calibrate.add_argument("landmarks_json", help="JSON with image_points and ground_points_m arrays")
    calibrate.add_argument("output_json")
    calibrate.add_argument("--max-rms-m", type=float, default=1.0)

    args = parser.parse_args()
    run_id = getattr(args, "run_id", None) or uuid.uuid4().hex[:10]
    if args.command == "prepare-risk-data":
        result = prepare_risk_data(args.dataset_root, args.output_csv, sample_every_s=args.sample_every_s,
                                   candidate_radius_m=args.candidate_radius_m)
        print(json.dumps(result, indent=2))
        return
    if args.command == "train-risk-model":
        result = train_risk_model(args.pair_windows_csv, args.model_out, metrics_out=args.metrics_out)
        print(json.dumps(result, indent=2))
        return
    if args.command == "build-network":
        build_osm_network(args.osm_file, args.output_net)
        print(f"Built SUMO network: {args.output_net}")
        return
    if args.command == "write-config":
        write_sumocfg(args.network, args.routes, args.output_cfg, end_s=args.duration)
        print(f"Wrote SUMO config: {args.output_cfg}")
        return
    if args.command == "seed-routes":
        config = json.loads(Path(args.paths_json).read_text(encoding="utf-8"))
        generate_seeded_routes(args.output_routes, vehicle_paths=config["vehicle_paths"],
                               pedestrian_paths=config["pedestrian_paths"], seed=args.seed,
                               demand=args.demand, conflict_mode=args.conflict_mode)
        print(f"Wrote seeded route file: {args.output_routes}")
        return
    if args.command == "sample-video-frames":
        result = sample_video_frames_for_annotation(args.video, args.output_dir, count=args.count)
        print(json.dumps(result, indent=2))
        return
    if args.command == "detection-metrics":
        print(json.dumps(detection_precision_recall(args.labels_csv, args.predictions_csv,
                                                      iou_threshold=args.iou,
                                                      evaluated_frames_csv=args.frames_csv), indent=2))
        return
    if args.command == "calibrate":
        landmarks = json.loads(Path(args.landmarks_json).read_text(encoding="utf-8"))
        result = fit_homography(landmarks["image_points"], landmarks["ground_points_m"],
                                maximum_rms_error_m=args.max_rms_m)
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote calibration with {result['landmark_count']} landmarks and {result['rms_error_m']:.3f} m RMS")
        return
    if args.command == "analyze":
        points = load_tracks(args.input_csv)
        if args.source_type != "sumo_gt":
            points = [type(p)(**{**p.__dict__, "source_type": args.source_type}) for p in points]
        zones = json.loads(Path(args.zones_json).read_text(encoding="utf-8")) if args.zones_json else None
        detection_metrics = None
        if args.detection_labels and args.detection_predictions:
            detection_metrics = detection_precision_recall(
                args.detection_labels, args.detection_predictions,
                evaluated_frames_csv=args.detection_frames_csv)
        run_dir = analyze_run(points, args.out, run_id=run_id, source_type=args.source_type,
                              max_ttc_s=args.max_ttc_s, max_pet_s=args.max_pet_s,
                              cell_size_m=args.cell_size_m, zones=zones,
                              detection_metrics=detection_metrics,
                              expected_events_csv=args.expected_events_csv,
                              risk_model_path=args.risk_model)
        print(f"Analyzed {len(points)} observations; results: {run_dir}")
        return
    if args.command == "simulate":
        paths = run_sumo(args.sumocfg, Path(args.out) / "work" / run_id)
        points = read_fcd(paths["fcd"], source_id=run_id, network_path=args.network)
        if args.render and points:
            evidence = Path(args.out) / "runs" / run_id / "evidence_safe"
            render_track_frames(points, evidence)
            if args.weights:
                calibration = json.loads((evidence / "render_calibration.json").read_text(encoding="utf-8"))
                detected = detect_and_track(evidence / "sumo_render.mp4", args.weights,
                                            source_id=run_id, source_type="sumo_detected",
                                            homography=calibration,
                                            synthetic_evidence_output=evidence / "sumo_detected_overlay.mp4",
                                            predictions_csv=Path(args.out) / "runs" / run_id / "detection_predictions.csv")
                points.extend(detected)
        ssm_rows = read_ssm_values(paths["ssm"])
        detection_metrics = None
        prediction_file = Path(args.out) / "runs" / run_id / "detection_predictions.csv"
        if args.detection_labels and prediction_file.exists():
            detection_metrics = detection_precision_recall(
                args.detection_labels, prediction_file,
                evaluated_frames_csv=args.detection_frames_csv)
        source_label = "sumo_gt+sumo_detected" if any(p.source_type == "sumo_detected" for p in points) else "sumo_gt"
        run_dir = analyze_run(points, args.out, run_id=run_id, source_type=source_label,
                              ssm_rows=ssm_rows, detection_metrics=detection_metrics)
        print(f"Simulation complete; results: {run_dir}; SSM: {paths['ssm']}")
        return
    if args.command == "detect":
        homography_data = json.loads(Path(args.homography_json).read_text(encoding="utf-8")) if args.homography_json else None
        homography = homography_data.get("homography", homography_data) if isinstance(homography_data, dict) else homography_data
        predictions_path = Path(args.out) / "runs" / run_id / "detection_predictions.csv"
        points = detect_and_track(args.video, args.weights, source_id=args.source_id,
                                  source_type="video", homography=homography,
                                  predictions_csv=predictions_path)
        detection_metrics = (detection_precision_recall(
            args.detection_labels, predictions_path,
            evaluated_frames_csv=args.detection_frames_csv) if args.detection_labels else None)
        run_dir = analyze_run(points, args.out, run_id=run_id, source_type="video",
                              detection_metrics=detection_metrics)
        print(f"Detection complete; results: {run_dir}")


if __name__ == "__main__":
    main()
