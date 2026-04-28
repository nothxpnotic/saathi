"""
====================================================================
SAATHI — Smart AI-Assisted Triage and Healthcare Interface
====================================================================
Supporting Evidence: Gradient Boosted Classifier (GBC) Pipeline
Author  : Muhammad Aayan Malik et al.
Org     : Nishtar Medical University, Multan, Pakistan
Purpose : Train, evaluate, and persist the offline triage model
          that powers the SAATHI OPD triage tablet application.
====================================================================

HOW TO RUN
----------
    pip install -r requirements.txt
    python saathi_model.py

The script will:
  1. Generate a synthetic training dataset (proxy for anonymised OPD
     registers — replace with real data before clinical validation).
  2. Preprocess and encode features.
  3. Train a Gradient Boosted Classifier.
  4. Print a full evaluation report (accuracy, AUC, confusion matrix).
  5. Save the trained model to  models/saathi_gbc_model.pkl
     and the label encoder to  models/label_encoder.pkl
====================================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────

TRIAGE_LABELS  = ["Routine", "Urgent", "Immediate"]   # 0, 1, 2
N_SAMPLES      = 5_000   # synthetic rows; replace with real OPD data
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "saathi_gbc_model.pkl")
ENCODER_PATH   = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")

# Chief-complaint categories (Urdu transliteration mapped to int code)
COMPLAINTS = {
    "chest_pain":          0,
    "shortness_of_breath": 1,
    "high_fever":          2,
    "abdominal_pain":      3,
    "head_injury":         4,
    "altered_sensorium":   5,
    "vomiting_diarrhoea":  6,
    "laceration_wound":    7,
    "general_weakness":    8,
    "routine_followup":    9,
}

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────
# NOTE: This synthetic dataset is used ONLY as a proof-of-concept.
#       Before any clinical validation, replace generate_dataset()
#       with a loader that reads from your anonymised OPD CSV/SQL.
# ─────────────────────────────────────────────────────────────────

def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Produce a physiologically plausible synthetic OPD dataset.

    Triage logic mirrors standard Emergency Severity Index (ESI)
    adapted for Pakistan OPD contexts:

    IMMEDIATE  — SpO2 < 90 | SBP < 80 | SBP > 180 | HR > 150
                 | HR < 40 | Temp > 40.5 | complaint in
                 {chest_pain, altered_sensorium, head_injury}

    URGENT     — SpO2 90–94 | SBP 80–90 | SBP 160–180 | HR 110–150
                 | HR 40–50 | Temp 38.5–40.5 | complaint in
                 {shortness_of_breath, high_fever, abdominal_pain}

    ROUTINE    — all remaining presentations
    """

    records = []
    complaint_keys = list(COMPLAINTS.keys())

    for _ in range(n):
        # ── vital signs drawn from realistic OPD distributions ──
        spo2     = np.clip(np.random.normal(97, 4),  60, 100)
        sbp      = np.clip(np.random.normal(120, 25), 60, 220)
        dbp      = np.clip(np.random.normal(80, 15),  40, 130)
        temp     = np.clip(np.random.normal(37.2, 1.2), 34, 42)
        hr       = np.clip(np.random.normal(82, 22),   35, 200)
        complaint_idx = np.random.choice(len(complaint_keys))
        complaint = complaint_keys[complaint_idx]
        c_code   = COMPLAINTS[complaint]

        # ── deterministic triage assignment ──
        if (
            spo2 < 90
            or sbp < 80 or sbp > 180
            or hr  > 150 or hr < 40
            or temp > 40.5
            or complaint in ("chest_pain", "altered_sensorium", "head_injury")
        ):
            label = "Immediate"

        elif (
            90 <= spo2 < 94
            or 80 <= sbp <= 90 or 160 <= sbp <= 180
            or 110 <= hr <= 150 or 40 <= hr <= 50
            or 38.5 <= temp <= 40.5
            or complaint in ("shortness_of_breath", "high_fever", "abdominal_pain")
        ):
            label = "Urgent"

        else:
            label = "Routine"

        records.append({
            "spo2":        round(spo2, 1),
            "sbp":         round(sbp,  1),
            "dbp":         round(dbp,  1),
            "temperature": round(temp, 1),
            "heart_rate":  round(hr,   1),
            "complaint":   c_code,
            "triage":      label,
        })

    df = pd.DataFrame(records)

    # Light noise injection: flip ~3 % labels to simulate real-world
    # annotation uncertainty in OPD registers
    noise_idx = df.sample(frac=0.03, random_state=1).index
    df.loc[noise_idx, "triage"] = np.random.choice(TRIAGE_LABELS, size=len(noise_idx))

    return df


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Encode the target label, separate features and target,
    and scale numeric vitals.

    Returns
    -------
    X_scaled : np.ndarray  — normalised feature matrix
    y        : np.ndarray  — integer-encoded triage category
    le       : LabelEncoder
    scaler   : StandardScaler
    feature_names : list[str]
    """
    le = LabelEncoder()
    le.classes_ = np.array(TRIAGE_LABELS)       # fix class order
    y = le.transform(df["triage"].values)

    feature_cols = ["spo2", "sbp", "dbp", "temperature", "heart_rate", "complaint"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────

def build_model() -> GradientBoostingClassifier:
    """
    Return a tuned GradientBoostingClassifier.

    Hyperparameters chosen for:
      • Good generalisation on small-to-medium OPD datasets
      • Fast inference on low-cost Android hardware (via ONNX export)
      • Stable probability calibration (for AUC reporting)
    """
    return GradientBoostingClassifier(
        n_estimators      = 300,
        learning_rate     = 0.08,
        max_depth         = 4,
        min_samples_split = 20,
        min_samples_leaf  = 10,
        subsample         = 0.8,
        max_features      = "sqrt",
        random_state      = 42,
        verbose           = 0,
    )


# ─────────────────────────────────────────────────────────────────
# SECTION 5 — EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, le, feature_names):
    """
    Print classification report, AUC, and save confusion-matrix plot.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n" + "═" * 60)
    print("  SAATHI — GRADIENT BOOSTED CLASSIFIER  EVALUATION REPORT")
    print("═" * 60)
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                digits=4))

    # Multi-class AUC (One-vs-Rest macro average)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    print(f"  Macro-average ROC-AUC (OvR) : {auc:.4f}")
    print("═" * 60)

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("SAATHI GBC — Confusion Matrix (Test Set)")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved → {cm_path}")

    # ── Feature importance ────────────────────────────────────────
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names,
                            "importance": importances})
    feat_df.sort_values("importance", ascending=True, inplace=True)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars = ax2.barh(feat_df["feature"], feat_df["importance"],
                    color="#1a6b9a")
    ax2.set_xlabel("Mean Decrease in Impurity")
    ax2.set_title("SAATHI GBC — Feature Importances")
    ax2.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved → {fi_path}\n")

    return auc


def cross_validate(model, X, y):
    """5-fold stratified cross-validation on the full dataset."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="roc_auc_ovr_weighted", n_jobs=-1)
    print(f"  5-Fold CV Weighted AUC: {scores.mean():.4f} "
          f"(± {scores.std():.4f})")


# ─────────────────────────────────────────────────────────────────
# SECTION 6 — INFERENCE HELPER (used by the UI)
# ─────────────────────────────────────────────────────────────────

def predict_triage(spo2: float, sbp: float, dbp: float,
                   temperature: float, heart_rate: float,
                   complaint_code: int,
                   model_path: str = MODEL_PATH,
                   scaler_path: str = SCALER_PATH,
                   encoder_path: str = ENCODER_PATH) -> dict:
    """
    Load the persisted model and return a triage prediction.

    Parameters
    ----------
    spo2, sbp, dbp, temperature, heart_rate : float — patient vitals
    complaint_code : int — integer code from COMPLAINTS dict

    Returns
    -------
    dict with keys:
        "label"       : str   — "Routine" | "Urgent" | "Immediate"
        "confidence"  : float — probability of predicted class (0–1)
        "probabilities": dict — full class probability breakdown
        "flag"        : str   — advisory flag for high-risk patterns
    """
    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    le      = joblib.load(encoder_path)

    X_raw   = np.array([[spo2, sbp, dbp, temperature, heart_rate, complaint_code]])
    X_scaled = scaler.transform(X_raw)

    pred_idx = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]
    label    = le.inverse_transform([pred_idx])[0]

    # High-risk flag logic (rule-based safety net on top of ML)
    flags = []
    if spo2 < 90:           flags.append("⚠ Critical SpO₂ — consider oxygen therapy")
    if sbp < 80:            flags.append("⚠ Hypotensive — urgent BP management")
    if sbp > 180:           flags.append("⚠ Hypertensive crisis — immediate review")
    if heart_rate > 150:    flags.append("⚠ Tachycardia — arrhythmia screen")
    if heart_rate < 40:     flags.append("⚠ Bradycardia — cardiac monitoring")
    if temperature > 40.5:  flags.append("⚠ Hyperpyrexia — antipyretics + workup")

    return {
        "label":         label,
        "confidence":    round(float(proba[pred_idx]), 4),
        "probabilities": {le.classes_[i]: round(float(p), 4)
                          for i, p in enumerate(proba)},
        "flags":         flags if flags else ["No critical flags detected"],
    }


# ─────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  SAATHI — Model Training Pipeline")
    print("═" * 60)

    # 1. Generate / load dataset
    print("\n[1/5] Generating synthetic OPD dataset …")
    df = generate_dataset(N_SAMPLES)
    print(f"      {len(df)} records | Label distribution:\n"
          f"{df['triage'].value_counts().to_string()}\n")

    # 2. Preprocess
    print("[2/5] Preprocessing features …")
    X, y, le, scaler, feature_names = preprocess(df)

    # 3. Train / test split (stratified to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 4. Cross-validate first
    print("\n[3/5] Running 5-fold cross-validation …")
    model_cv = build_model()
    cross_validate(model_cv, X_train, y_train)

    # 5. Final fit on full training split
    print("\n[4/5] Training final model on full training set …")
    model = build_model()
    model.fit(X_train, y_train)
    print("      Training complete.")

    # 6. Evaluate on held-out test set
    print("\n[5/5] Evaluating on held-out test set …")
    auc = evaluate(model, X_test, y_test, le, feature_names)

    # 7. Persist artefacts
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)
    print(f"  Model   saved → {MODEL_PATH}")
    print(f"  Scaler  saved → {SCALER_PATH}")
    print(f"  Encoder saved → {ENCODER_PATH}")

    print("\n  ✓ SAATHI model pipeline complete.")
    print(f"  Final Test AUC: {auc:.4f}")
    print("═" * 60 + "\n")

    # 8. Quick inference demo
    print("  Sample Inference Demo:")
    print("  Input: SpO2=88, SBP=170, DBP=100, Temp=37.2, HR=105, "
          "Complaint=chest_pain")
    result = predict_triage(
        spo2=88, sbp=170, dbp=100,
        temperature=37.2, heart_rate=105,
        complaint_code=COMPLAINTS["chest_pain"]
    )
    print(f"  → Triage Label  : {result['label']}")
    print(f"  → Confidence    : {result['confidence']:.2%}")
    print(f"  → Probabilities : {result['probabilities']}")
    print(f"  → Clinical Flags: {result['flags']}\n")


if __name__ == "__main__":
    main()
