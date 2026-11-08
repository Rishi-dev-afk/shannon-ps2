# Codebase File Guide

This guide describes the files in the PS-02 Vision Zero project and how they fit together. The Python package contains the analysis pipeline; the Streamlit app displays results already written by that pipeline.

## Project root

- `.gitignore` — Keeps generated outputs, local environments, model weights, videos, logs, and other local-only or sensitive artifacts out of Git. The local WhatsApp screenshot is explicitly excluded.
- `LICENSE` — The repository's MIT license, which states the terms for using, copying, modifying, and distributing the project. (This file is present on GitHub.)
- `README.md` — Project overview, quick-start commands, input formats, workflows, output descriptions, and current limitations.
- `FILE_GUIDE.md` — This map of the project's files and their roles.
- `pyproject.toml` — Package metadata, supported Python versions, pinned dependencies, package discovery settings, and the `vision-zero` command entry point.
- `requirements.txt` — Pinned Python dependencies for installing the runtime libraries.
- `run_demo.ps1` — PowerShell helper for installing the package, generating deterministic sample tracks, running an analysis, and starting the dashboard.

## `src/vision_zero/` — Analysis package

- `__init__.py` — Marks `vision_zero` as an importable Python package.
- `schema.py` — Shared data records for track observations, candidate risk events, and hotspot summaries.
- `cli.py` — Defines the `vision-zero` command line interface: analyze tracks, prepare SUMO networks/configs/routes, simulate, detect video tracks, calibrate a camera transform, prepare/train the optional RF model, sample video frames for annotation, and score detections.
- `pipeline.py` — Connects ingestion-ready track records to TTC/PET event screening, validation, hotspot aggregation, report writing, and versioned run outputs.
- `ingest.py` — Reads and normalizes track CSV files, maps road-user classes into groups, and estimates velocities while flagging track-quality issues.
- `indicators.py` — Detects candidate time-to-collision (TTC) events from short-horizon motion estimates and classifies conflict types and severity bands.
- `pet.py` — Estimates post-encroachment time (PET) for road users traversing configured shared conflict zones.
- `hotspots.py` — Groups candidate events into metric grid cells and calculates counts, exposure context, rates, rough intervals, and dominant patterns.
- `validation.py` — Compares event sets and data sources, reports threshold sensitivity and SUMO SSM agreement, and can compare results with expected seeded events.
- `detection_metrics.py` — Scores hand-checked object detections using bounding-box intersection-over-union, precision, and recall; it also samples source-video frames and reads the reviewed-frame manifest.
- `sumo.py` — Wraps local SUMO tools to build networks, write simulation configuration, run simulations, and read FCD tracks and SSM encounters.
- `scenario.py` — Writes repeatable SUMO route files for crossing, turning, bike cut-across, or control scenarios using supplied connected edge paths.
- `vision.py` — Renders synthetic tracks into video and optionally runs Ultralytics detection/tracking to produce track records and prediction outputs.
- `calibration.py` — Fits an image-to-road-plane homography from paired image and surveyed ground landmarks and reports its fit error.
- risk_model.py — Prepares causal trajectory-pair windows with Knowledge PET3D source-method proxy labels, performs location-held-out Random Forest evaluation, saves the trained artifact/schema, and optionally scores metric-coordinate analysis tracks.
- `report.py` — Writes run tables and engineer-facing summaries in CSV, GeoJSON, Markdown, HTML, and PDF formats.
- `privacy.py` — Checks generated/displayed text for restricted blame language and verifies that dashboard evidence files stay inside the designated safe-evidence folder.


## docs/ — Data and evaluation guides

- KNOWLEDGE_PET3D_PROVENANCE.md — Dataset attribution, local input layout, proxy-label definition, and RF preparation/training commands.
- VIDEO_DETECTION_EVALUATION.md — External source-video sampling, hand-labeling, and detector scoring workflow; records the limits of current two-frame evidence.
- IMPLEMENTATION_CHANGELOG.md — Change-by-change record of the RF/detector plan implementation, verification status, remaining data dependencies, and Colab Drive extraction script.


## `app/` — Streamlit review dashboard

- `Home.py` — Dashboard landing page with the selected run's headline counts, quality note, and report navigation.
- `lib/data.py` — Finds and loads precomputed run bundles, selects the active run, checks displayed text, and shows source/run context.
- `pages/1_Detection.py` — Reviews tracks and displays only pre-rendered evidence from the run's `evidence_safe` folder.
- `pages/2_Event_Explorer.py` — Filters candidate events, explains the selected indicator, and plots involved tracks when coordinates are available.
- `pages/3_Hotspot_Map.py` — Displays geographic hotspot GeoJSON when available and otherwise presents metric grid-cell results for review.
- `pages/4_Patterns.py` — Summarizes conflict types, road-user pairs, severity, event timing, and threshold sensitivity.
- `pages/5_Validation.py` — Displays the run's validation summary.
- `pages/6_Method.py` — Explains the screening methods, coordinate requirements, coverage caveats, and intended engineering interpretation.
- `pages/7_Report_Export.py` — Provides links to the generated CSV, GeoJSON, Markdown, HTML, and PDF report files.
- `pages/8_What_If.py` — Compares event counts between two precomputed runs, such as a baseline and a changed-design scenario.

## `examples/` — Small inputs and demo data

- `generate_sample.py` — Creates deterministic synthetic track observations for a seeded conflict and a control movement.
- `osm_paths.template.json` — Placeholder JSON structure for vehicle and pedestrian SUMO edge paths; replace the placeholders with connected edge IDs from the chosen local network.

## `ps/` — Problem statement planning and input guidance

- `PS-02-Vision-Zero-Updated-Plan-v3.md` — The active project plan: objectives, scope, system design, methods, validation approach, edge cases, and delivery plan.
- `PS-02-Vision-Zero-Input-Setup-Guide.md` — Where to obtain required inputs, how to prepare them, and local versus hosted compute/setup guidance.
- `Screenshot (553).png` — A local reference screenshot that is ignored by Git because it contains a visible phone number. It is not included in the GitHub repository.

## How the files work together

1. `examples/generate_sample.py` or a user-provided source creates track data. For SUMO and video workflows, `cli.py` calls the adapters in `sumo.py` and `vision.py` first.
2. `cli.py` passes normalized track records into `pipeline.py`. `ingest.py` reads CSV tracks; `schema.py` defines the shared record shapes.
3. `pipeline.py` runs the indicator, hotspot, and validation modules, then calls `report.py` to write a run bundle under `outputs/runs/<run_id>/`.
4. The Streamlit pages use `app/lib/data.py` to read that saved bundle. The dashboard does not run simulation or model inference live.
5. `README.md` and the two planning/setup documents describe how to configure inputs, run the workflow, and interpret its limits.

SUMO binaries, OSM extracts, camera footage, trained weights, calibration landmarks, and Knowledge PET3D source files and hand-checked labels are user-supplied local inputs. Generated run folders and model/video files are not source files and are excluded or kept local according to `.gitignore`.
