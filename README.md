# PS-02 Vision Zero

Offline-first starter for the active [v3 project plan](ps/PS-02-Vision-Zero-Updated-Plan-v3.md). It includes track ingestion, TTC/PET screening, scenario generation helpers, SUMO FCD/SSM adapters, optional YOLO tracking, validation summaries, hotspot exports, neutral reports, and a precomputed-results Streamlit dashboard.

For download sources, file locations, calibration, and setup commands, see the [Input and Setup Guide](ps/PS-02-Vision-Zero-Input-Setup-Guide.md).

## Run the local demo

Use Python 3.11 or 3.12. From PowerShell:

```powershell
.
run_demo.ps1
```

Or run the stages directly:

```powershell
python -m pip install -e .
python examples/generate_sample.py
vision-zero analyze examples/sample_tracks.csv --out outputs --run-id seeded-demo
streamlit run app/Home.py
```

The sample tracks are synthetic and deterministic. Analysis writes one versioned bundle to `outputs/runs/<run_id>/` with tracks, events, hotspots, validation JSON, configuration, and HTML/PDF/Markdown report files. The dashboard reads those outputs and does not run SUMO or inference live.

Run the regression tests with `python -m pytest tests` (needs `pip install pytest`).

TTC events are collapsed to one event per encounter (the minimum-TTC sample of consecutive detections of the same pair; a new encounter starts after a 1 s gap). Pass `collapse=False` to `detect_ttc_events` for the raw per-timestep series.

## Deploy the dashboard

The hosted app is read-only: it displays the precomputed run(s) committed under `outputs/runs/<run_id>/` and never runs detection or SUMO. `requirements.txt` installs only the dashboard dependencies (no PyTorch, OpenCV or SUMO). Use `requirements-pipeline.txt` for the offline pipeline.

**Streamlit Community Cloud:** new app, main file `app/Home.py`, Advanced settings -> Python 3.12.

**Render:** push the repo and create a Blueprint from `render.yaml`, or create a Web Service manually with build command `pip install -r requirements.txt` and start command `streamlit run app/Home.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`. Set `PYTHON_VERSION` to `3.12.3`. Free instances sleep when idle, so open the URL a few minutes before a demo.

**Before you push:** add your final run folder under `outputs/runs/`, keep `evidence_raw/`, raw footage and `*.pt` weights out of git, and confirm everything in `evidence_safe/` is blurred or synthetic.

**Detector run:** `finetune_spike.py` fine-tunes weights on rendered frames. Then run `vision-zero detect ... --weights <best.pt> --homography-json render_calibration.json` and `vision-zero analyze` to produce a detected-track run to commit alongside the ground-truth one.

## SUMO and OSM workflow

Install Eclipse SUMO separately and make `sumo`, `netconvert`, and optionally `sumo-gui` available on `PATH`. Obtain a permitted local OSM XML/PBF extract for the selected intersection. SUMO builds the road network locally; no raw real-world video belongs in the repository.

1. Convert the OSM extract: `vision-zero build-network area.osm network.net.xml`.
2. Copy `examples/osm_paths.template.json`, replacing placeholders with connected vehicle and pedestrian edge paths from that network.
3. Generate repeatable conflict routes: `vision-zero seed-routes paths.json conflict.rou.xml --seed 42 --conflict-mode crossing`. Generate a no-conflict control with `--conflict-mode control`.
4. Create a SUMO config: `vision-zero write-config network.net.xml conflict.rou.xml scenario.sumocfg --duration 180`.
5. Simulate, render, and analyze ground truth: `vision-zero simulate scenario.sumocfg --network network.net.xml --render --out outputs --run-id crossing-seed-42`.
6. To run the detector on the rendered synthetic video too, add `--weights yolo26n.pt`. The resulting GT and detected tracks are analyzed separately and compared.

The route generator supports `crossing`, `turning`, `bike_cut_across`, and `control` seeds. SUMO's SSM output is kept beside the run and TTC/PET/DRAC values are matched to our events when the exported encounter IDs and measure values are available. The OSM network, connected edge paths, and SUMO installation are required inputs; this repository does not embed a third-party map extract.

## Video detection

Provide permitted footage and a compatible Ultralytics weight file locally:

```powershell
vision-zero detect clip.mp4 --weights yolo26n.pt --source-id clip-01 --out outputs --run-id clip-01
```

Without `--homography-json`, detections remain pixel-coordinate tracks and metric TTC/PET are intentionally suppressed. A homography JSON must be a 3×3 image-to-road-plane transform based on documented calibration landmarks. Detection output stores track metadata only; the dashboard only displays media saved in the `evidence_safe` directory. Redact faces and plates before placing evidence there.

Prepare calibration with surveyed point pairs in JSON (`image_points` and `ground_points_m`):

```powershell
vision-zero calibrate landmarks.json calibration.json --max-rms-m 1.0
vision-zero detect clip.mp4 --weights yolo26n.pt --source-id clip-01 --homography-json calibration.json --out outputs --run-id clip-01-calibrated
```

For hand-checked object detection metrics, prepare label and prediction CSVs with `frame_id,class,x1,y1,x2,y2` and run:

```powershell
vision-zero detection-metrics labels.csv predictions.csv --iou 0.5
```

## Track CSV contract

Required fields: `source_id,timestamp_s,track_id,class,x_m,y_m`. Optional fields include `source_type` (`sumo_gt`, `sumo_detected`, `video`), `class_group`, `coordinate_system`, `confidence`, `radius_m`, `quality_flags`, `lat`, and `lon`.

Coordinates must be a shared metric road-plane system for metric indicators. Pixel-only tracks are retained for detection review, but metric event analysis is skipped. Duplicate or missing samples and long time gaps should be flagged in `quality_flags`.

## Pipeline and outputs

- **Scenario:** local OSM network, route seeds, SUMO simulation, FCD tracks, and SSM encounters.
- **Vision:** optional YOLO detector with ByteTrack; video points are calibrated only when a homography is supplied.
- **Indicators:** constant-velocity buffered-footprint TTC, conflict-zone PET, and a supporting DRAC estimate. Thresholds and assumptions appear in each run config/report.
- **Validation:** GT-versus-detected event matches, spurious counts, threshold sensitivity, and available SUMO SSM agreement.
- **Hotspots:** metric grid cells with event counts, observed road users, exposure-normalized rates, rough rate intervals, and coverage flags. GeoJSON includes events only when latitude/longitude is available.
- **Privacy and neutral language:** no face/plate recognition is implemented. Text scanning rejects restricted blame terms. Only pre-redacted files under `evidence_safe` are served by the dashboard.
- **Dashboard:** Overview, Detection, Event Explorer, Hotspot Map, Patterns, Validation, Method, Report & Export, and What-if pages.

## Known integration limits

SUMO binaries, an OSM extract, connected edge paths, detector weights, a permitted real clip, camera calibration, and hand-checked labels are external inputs. The route generator cannot infer valid intersection edge paths automatically. SUMO's encounter output schemas depend on its version; the parser preserves encounter values and comparison is reported only for matched IDs/measurements. Hotspots use metric grid cells until road-segment geometry is supplied. PET is sampled at observed timestamps, so the zone-entry/exit time is approximate.


## Random Forest source-method screening

The optional pipeline extra includes scikit-learn. To use the Knowledge PET3D Random Forest, download/extract only the trajectory and published conflict CSVs for Location A and Location B outside Git, then follow [the data provenance and training guide](docs/KNOWLEDGE_PET3D_PROVENANCE.md).

The classifier's label is whether the published source method flagged an interaction in the next two seconds. “Not flagged” is not a confirmed safe example or a crash prediction. Existing TTC/PET rule analysis remains available without a model; pass --risk-model to analyze only after training an artifact. The raw dataset and videos stay outside the repository.

## Video detector evaluation

For the permitted original clip, the 100-frame annotation and IoU 0.5 scoring workflow is documented in [Video detector evaluation](docs/VIDEO_DETECTION_EVALUATION.md). The committed clip-01 evidence video is annotated output, so it is not an independent detector input.
