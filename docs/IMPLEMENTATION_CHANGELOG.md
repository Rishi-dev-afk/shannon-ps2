# Implementation change log: model and evaluation plan

## Repository state

- Checkout: C:\shannon-ps2
- Feature branch: codex/vision-zero-ml
- Base: updated main at commit c9f2c06
- Changes are local and uncommitted; this feature branch has not been pushed to GitHub.

## Changes made

| Area | Files | What changed |
|---|---|---|
| PET3D data preparation | src/vision_zero/risk_model.py, src/vision_zero/cli.py | Added prepare-risk-data. It reads the Location A/B trajectory and conflict CSVs, forms pair-window examples using the preceding 1 second, and labels whether the source method flagged a conflict in the following 2 seconds. Feature construction uses only observations at or before the window end. PET, LOF, source conflict scores/flags, IDs, and location are excluded from features. |
| Random Forest training | src/vision_zero/risk_model.py, src/vision_zero/cli.py | Added train-risk-model with 300 trees, min_samples_leaf=5, balanced_subsample class weighting, and seed 42. Evaluation holds out Location A then Location B; threshold tuning uses separate recording groups within the training location. It reports PR-AUC, precision, recall, F1, confusion counts, and held-out permutation feature importance. The trained artifact is saved compressed with a feature schema and evaluation JSON. |
| Optional inference | src/vision_zero/pipeline.py, src/vision_zero/risk_model.py | Added optional --risk-model scoring to analysis. Rule-only analysis remains unchanged. Metric-coordinate pair scores are exported as model_scores.csv with the pair, window times, score, decision, model version, and proxy-label source metadata. Pixel-only tracks are not scored by this metric-coordinate model. |
| Validation/reporting | app/pages/5_Validation.py, src/vision_zero/report.py | Added held-out RF metrics, both location folds, confusion counts, feature importance, and proxy-label caveats to the Validation page and engineer report. The report explains that these proxy-label metrics are not directly comparable with TTC/PET event-matching metrics. |
| Detector evaluation | src/vision_zero/detection_metrics.py, src/vision_zero/cli.py | Added sample-video-frames to export evenly spaced source frames, a frame manifest, and a label template. Detection scoring now counts predictions only on reviewed frames; a manifest can include reviewed frames with no objects. Car, bus, truck, and van predictions are grouped as vehicle for the current label scheme. |
| Dependencies and guides | pyproject.toml, README.md, FILE_GUIDE.md, docs/KNOWLEDGE_PET3D_PROVENANCE.md, docs/VIDEO_DETECTION_EVALUATION.md | Added scikit-learn to the optional pipeline extra only. Added dataset attribution/provenance, local folder layout, proxy-label explanation, the Colab source-video labeling workflow, and file-role documentation. |
| Regression coverage | tests/test_risk_model.py | Added checks for causal features, forward label alignment, timestamp gaps, missing tracks, raw CSV preparation, imbalanced classes, location-held-out groups, model persistence, score export, and reviewed-frame detector metrics. |

## Label interpretation

The RF target is whether Knowledge PET3D's published method flagged a pair interaction in the next two seconds. A zero means only “not flagged by that source method”; it is not a confirmed safe example. The classifier does not predict crashes.

The source dataset and its published analysis code are available from the [Mendeley dataset record](https://data.mendeley.com/datasets/2wtbb9jnpx/1) and the [official Knowledge PET3D repository](https://github.com/4rnd25/knowledge_PET3D). The dataset record lists CC BY 4.0. Keep attribution with any redistributed derived data. Do not put the large source archive, raw video, or detector weights in Git.

## Verification completed

- Python compilation passed for src, app, and tests.
- Focused RF checks passed: causal past-only features, next-two-second target alignment, timestamp-gap handling, missing-track handling, imbalanced training, model save/load, held-out feature importance, score CSV export, and analysis result integration.
- Raw per-record trajectory/conflict CSV preparation passed on a small generated fixture.
- Existing rule-only pipeline regression checks passed.
- Detector metric check at IoU 0.5 correctly limited scoring to the two labeled frames currently in the repository: vehicle precision 0.724 / recall 0.618; person precision 1.000 / recall 0.200.
- Pytest is not installed in the available environment, so these checks were invoked directly. The available interpreter is Python 3.13, while this repository declares Python 3.11 to 3.12; a run in the declared interpreter range is still needed.

## Work that still needs external inputs

- No real Knowledge PET3D source files were present in the checkout, so a real trained model artifact and its real held-out scores have not been generated.
- The original permitted video is not present. The committed clip-01 video is annotated YOLOv4 Darknet fallback output, not independent detector input. Current labels cover only two frames, not the planned 100-frame YOLO26n + ByteTrack evaluation.
- The implementation branch is local and unpushed. A Colab notebook cloned from GitHub will not see these new commands until the branch is pushed or the changes are otherwise copied into that runtime.

## Accessing the dataset from Colab

1. Open the [Knowledge PET3D Mendeley record](https://data.mendeley.com/datasets/2wtbb9jnpx/1), download its archive, and upload the ZIP to a folder in Google Drive, for example My Drive/datasets/Knowledge_PET3D.zip.
2. In Colab, run the first cell below and authorize Drive access.
3. Run the extraction cell. It extracts only CSV files beneath Location A/B Trajectories or Conflicts folders into the temporary Colab runtime. It checks available disk space first.
4. Colab runtime files are temporary. Keep the ZIP in Drive so it can be extracted again after a runtime reset.

Cell 1 — mount Drive:

    from google.colab import drive
    drive.mount("/content/drive")

Cell 2 — locate the ZIP and extract only the trajectory/conflict CSVs:

    from pathlib import Path
    import shutil
    import zipfile

    zip_path = Path("/content/drive/MyDrive/datasets/Knowledge_PET3D.zip")
    output_root = Path("/content/knowledge_pet3d")

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"ZIP not found at {zip_path}. Upload the Mendeley download to that Drive folder "
            "or update zip_path."
        )

    def is_required_csv(member_name):
        normalized = member_name.replace("\\", "/")
        parts = normalized.lower().split("/")
        is_csv = normalized.lower().endswith(".csv")
        is_location = any("location_a" in part or "location_b" in part for part in parts)
        is_target_folder = any(part in {"trajectories", "conflicts"} for part in parts[:-1])
        return is_csv and is_location and is_target_folder

    with zipfile.ZipFile(zip_path) as archive:
        selected = [name for name in archive.namelist() if is_required_csv(name)]
        if not selected:
            csv_examples = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ][:20]
            raise RuntimeError(
                "No trajectory/conflict CSVs matched. First CSV paths in archive: "
                + repr(csv_examples)
            )

        required_bytes = sum(archive.getinfo(name).file_size for name in selected)
        free_bytes = shutil.disk_usage("/content").free
        print(f"CSV files selected: {len(selected)}")
        print(f"Uncompressed size: {required_bytes / 1e9:.2f} GB")
        print(f"Free Colab space: {free_bytes / 1e9:.2f} GB")
        if free_bytes < required_bytes + 500_000_000:
            raise RuntimeError(
                "Not enough free space in /content. Free space or use a runtime with more disk."
            )

        for name in selected:
            archive.extract(name, output_root)

    print("Extracted data root:", output_root)
    print("Sample extracted files:")
    for path in list(output_root.rglob("*.csv"))[:10]:
        print(path)

## Prepare and train in Colab after the feature code is available there

The new CLI commands are not available in a Colab clone of GitHub main until codex/vision-zero-ml is pushed or copied to that runtime. This checkout requires Python 3.11 or 3.12; check the runtime version before installing the pipeline extra.

After syncing the feature branch and installing the supported environment:

    %cd /content/shannon-ps2
    !python -m pip install -e ".[pipeline]"
    !python -m vision_zero.cli prepare-risk-data /content/knowledge_pet3d /content/drive/MyDrive/vision-zero/knowledge-pet3d-pair-windows.csv
    !python -m vision_zero.cli train-risk-model /content/drive/MyDrive/vision-zero/knowledge-pet3d-pair-windows.csv --model-out /content/drive/MyDrive/vision-zero/knowledge-pet3d-rf.joblib

These commands keep the prepared windows, model, schema, and evaluation metrics in Drive. The raw archive and source videos remain outside the Git repository.
