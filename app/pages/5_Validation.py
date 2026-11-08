import pandas as pd
import streamlit as st

from lib.data import current_run, source_banner
from lib.ui import STRETCH, page, pretty

page("Validation", "✅", "How well do screening results agree with references?")
run_id, run = current_run()
if not run_id:
    st.stop()
source_banner(run_id, run)

validation = run.get("validation", {})
if not validation:
    st.info("No validation summary is attached to this run yet.")
    st.stop()

if validation.get("quality_banner"):
    st.info(validation["quality_banner"], icon="ℹ️")

# --- Seeded scenarios -------------------------------------------------------
st.subheader("Seeded scenarios")
seeded = validation.get("seeded_scenarios", {})
if seeded:
    a, b, c, d = st.columns(4)
    a.metric("Expected events", seeded.get("seeded_expected_events", "n/a"))
    b.metric("Matched", seeded.get("matched_events", "n/a"))
    recall = seeded.get("event_recall")
    c.metric("Event recall", f"{recall:.0%}" if isinstance(recall, (int, float)) else "n/a")
    d.metric("Extra candidates", seeded.get("extra_candidate_events", "n/a"))
    tol = seeded.get("time_tolerance_s")
    if tol is not None:
        st.caption(f"An event counts as matched when it falls within {tol:g} s of an expected one. "
                   "Extra candidates are screening output with no seeded counterpart; review them rather than discard them.")
else:
    st.caption("This run has no seeded scenarios.")

# --- SUMO SSM -------------------------------------------------------------
st.subheader("Reference comparison (SUMO SSM)")
ssm = validation.get("ssm_comparison", {})
if not ssm or not ssm.get("ssm_encounters"):
    st.caption("No SUMO surrogate-safety encounters are available, so no TTC/PET agreement can be reported.")
else:
    a, b, c = st.columns(3)
    a.metric("SSM encounters", ssm["ssm_encounters"])
    matched = ssm.get("matched_events", {})
    b.metric("Matched TTC", matched.get("TTC", 0))
    c.metric("Matched PET", matched.get("PET", 0))
    mae = {k: v for k, v in ssm.get("mean_absolute_error_s", {}).items() if v is not None}
    if mae:
        st.dataframe(pd.DataFrame({"Indicator": list(mae), "Mean absolute error (s)": list(mae.values())}),
                     hide_index=True, **STRETCH)

# --- Detection ------------------------------------------------------------
st.subheader("Detector metrics")
det = validation.get("detection_metrics", {})
if det:
    if det.get("per_class"):
        st.caption(f"IoU threshold: {det.get('iou_threshold', 'n/a')}; reviewed frames: {det.get('evaluated_frame_count', 'n/a')}.")
        detector_table = pd.DataFrame.from_dict(det["per_class"], orient="index").rename_axis("Class").reset_index()
        st.dataframe(detector_table.rename(columns=pretty), hide_index=True, **STRETCH)
    else:
        st.dataframe(pd.DataFrame([det]).rename(columns=pretty), hide_index=True, **STRETCH)
else:
    st.caption("No hand-labeled detector comparison is attached to this run.")

# --- Random Forest proxy-label evaluation ----------------------------------
risk = validation.get("risk_model", {})
if risk:
    st.subheader("Random Forest: source-method flag screening")
    st.warning(
        "The model learns whether Knowledge PET3D's published method flags an interaction. "
        "A negative label means the source method did not flag it; it does not mean the interaction is safe.",
        icon="⚠️",
    )
    held_out = risk.get("held_out_evaluation", {})
    aggregate = held_out.get("aggregate_held_out_metrics", {})
    a, b, c, d = st.columns(4)
    pr_auc = held_out.get("held_out_pr_auc")
    a.metric("Held-out PR-AUC", f"{pr_auc:.3f}" if isinstance(pr_auc, (int, float)) else "n/a")
    for col, label in ((b, "Precision"), (c, "Recall"), (d, "F1")):
        value = aggregate.get(label.lower())
        col.metric(f"Held-out {label}", f"{value:.3f}" if isinstance(value, (int, float)) else "n/a")
    folds = held_out.get("folds", [])
    if folds:
        st.caption("Each row holds out one location; the threshold is selected on separate recording groups from the training location. These proxy-label scores are not directly comparable with TTC/PET event-matching metrics because they use a different target.")
        st.dataframe(pd.DataFrame(folds)[[
            "train_location", "test_location", "threshold", "pr_auc_average_precision",
            "precision", "recall", "f1", "confusion_counts",
        ]], hide_index=True, **STRETCH)
    importance = held_out.get("permutation_feature_importance", [])
    if importance:
        st.subheader("Held-out permutation feature importance")
        st.dataframe(pd.DataFrame(importance).head(15), hide_index=True, **STRETCH)
    if risk.get("score_note"):
        st.caption(risk["score_note"])
    else:
        st.caption(f"Scores exported for {risk.get('score_rows', 0)} pair windows. See model_scores.csv in this run's result folder.")

src = validation.get("source_comparison", {})
if src and src.get("comparisons"):
    st.subheader("Source comparison")
    st.json(src["comparisons"], expanded=False)

with st.expander("Raw validation JSON"):
    st.json(validation)
