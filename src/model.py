"""
model.py — Step 2: Classifier

Loads outputs/features.csv (FFT/ELA/noise/metadata signals from features.py),
merges in outputs/vgg_predictions_merged.csv (pretrained VGG16 probability,
from batch_predict.py) on filename, scales features, and evaluates two
classifiers using Leave-One-Out Cross-Validation (LOOCV) -- right choice at
small n (150), since a train/test split this small is close to meaningless.

Also supports outputs/video_features.csv the same way, if present.

Prints LOOCV accuracy, confusion matrix, precision/recall, logreg
coefficients, RF feature importances. Saves final fitted models +
per-row LOOCV predictions to outputs/.

Run from project root:
    python3 src/model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(PROJECT_ROOT, "outputs", "features.csv")
VGG_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "outputs", "vgg_predictions_merged.csv")
VIDEO_FEATURES_CSV = os.path.join(PROJECT_ROOT, "outputs", "video_features.csv")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "outputs", "predictions.csv")
VIDEO_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "outputs", "video_predictions.csv")
MODEL_ARTIFACT = os.path.join(PROJECT_ROOT, "outputs", "model_artifacts.joblib")
VIDEO_MODEL_ARTIFACT = os.path.join(PROJECT_ROOT, "outputs", "video_model_artifacts.joblib")

# Full feature set: original 4 forensic signals + metadata pre-filter
# (from features.py's metadata_suspicion_flags) + pretrained VGG16
# probability (from batch_predict.py, merged in at load time below).
FEATURE_COLS = [
    "fft_hf_ratio", "ela_variance", "ela_mean", "noise_perfection_score",
    "width", "height", "bytes_per_pixel", "suspicious_dims",
    "vgg_prob_real",
]
LABEL_COL = "label"

VIDEO_FEATURE_COLS = [
    "fft_hf_ratio_mean", "fft_hf_ratio_temporal_var",
    "ela_variance_mean", "ela_variance_temporal_var",
    "ela_mean_mean", "ela_mean_temporal_var",
    "noise_perfection_score_mean", "noise_perfection_score_temporal_var",
]


def load_data(csv_path, feature_cols):
    """
    Generic loader. If vgg_prob_real is in feature_cols, merges it in from
    VGG_PREDICTIONS_CSV on filename -- keeps features.py and batch_predict.py
    fully independent (neither needs to know about the other) while letting
    model.py combine both signal sources into one feature matrix.
    """
    df = pd.read_csv(csv_path)

    if "vgg_prob_real" in feature_cols:
        if os.path.exists(VGG_PREDICTIONS_CSV):
            vgg_df = pd.read_csv(VGG_PREDICTIONS_CSV)
            df = df.merge(vgg_df[["filename", "vgg_prob_real"]], on="filename", how="left")
            missing = df["vgg_prob_real"].isna().sum()
            if missing:
                print(f"  [!] {missing} row(s) missing vgg_prob_real (filename mismatch?) -- filling 0.5 (neutral)")
                df["vgg_prob_real"] = df["vgg_prob_real"].fillna(0.5)
        else:
            print(f"  [!] {VGG_PREDICTIONS_CSV} not found -- vgg_prob_real will be missing, filling 0.5")
            df["vgg_prob_real"] = 0.5

    X = df[feature_cols].values
    y = (df[LABEL_COL] == "fake").astype(int).values  # 1 = fake, 0 = real
    return df, X, y


def run_loocv(model_name, model_builder, X, y):
    """
    Leave-One-Out CV. model_builder is a zero-arg function returning a
    fresh, unfitted model each fold. Scaler refit per-fold -- no leakage.
    """
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_builder()
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)[0]
        proba = model.predict_proba(X_test_scaled)[0][1]  # P(fake)

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_proba.append(proba)

    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)

    acc = accuracy_score(y_true, y_pred)
    n_correct = int(np.sum(y_true == y_pred))
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} (LOOCV, n={len(y)}) ---")
    print(f"Accuracy:  {acc:.3f}  ({n_correct}/{len(y)} correct)")
    print(f"Precision (fake): {prec:.3f}")
    print(f"Recall (fake):    {rec:.3f}")
    print("Confusion matrix (rows=true, cols=predicted, order=[real, fake]):")
    print(cm)
    print("Full report (both classes):")
    print(classification_report(y_true, y_pred, target_names=["real", "fake"], zero_division=0))

    return y_pred, y_proba


def print_logreg_coefficients(X, y, feature_cols):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    print("\n--- Logistic Regression coefficients ---")
    print("(Positive = pushes toward 'fake', negative = pushes toward 'real')")
    for feat, coef in zip(feature_cols, model.coef_[0]):
        print(f"  {feat:32s} {coef:+.3f}")


def print_rf_importances(X, y, feature_cols):
    """Fits on RAW (unscaled) X -- tree splits are scale-invariant."""
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)

    print("\n--- Random Forest feature importances ---")
    for feat, imp in zip(feature_cols, model.feature_importances_):
        print(f"  {feat:32s} {imp:.3f}")


def train_and_save(csv_path, feature_cols, predictions_csv, model_artifact, label):
    """Full LOOCV + interpretability + final-fit-and-save pipeline for one
    feature set. Returns True if it ran, False if csv missing/empty."""
    if not os.path.exists(csv_path):
        print(f"\n[{label}] No file at {csv_path} -- skipping (optional).")
        return False

    df, X, y = load_data(csv_path, feature_cols)
    if len(df) == 0:
        print(f"\n[{label}] {csv_path} is empty -- skipping.")
        return False

    print(f"\n{'='*70}\n[{label}] Loaded {len(df)} rows | real: {(y==0).sum()} | fake: {(y==1).sum()}")
    print(f"[{label}] Features used: {feature_cols}")

    logreg_pred, logreg_proba = run_loocv(
        f"{label} - Logistic Regression",
        lambda: LogisticRegression(max_iter=1000),
        X, y
    )
    rf_pred, rf_proba = run_loocv(
        f"{label} - Random Forest (shallow)",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        X, y
    )

    print_logreg_coefficients(X, y, feature_cols)
    print_rf_importances(X, y, feature_cols)

    final_scaler = StandardScaler()
    X_scaled_all = final_scaler.fit_transform(X)

    final_logreg = LogisticRegression(max_iter=1000)
    final_logreg.fit(X_scaled_all, y)

    final_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    final_rf.fit(X, y)

    joblib.dump({
        "scaler": final_scaler,
        "logreg": final_logreg,
        "rf": final_rf,
        "feature_cols": feature_cols,
    }, model_artifact)
    print(f"\n[{label}] Saved final scaler + models to {model_artifact}")

    out_df = df[["filename", "label"]].copy()
    out_df["logreg_pred"] = np.where(logreg_pred == 1, "fake", "real")
    out_df["logreg_p_fake"] = logreg_proba
    out_df["rf_pred"] = np.where(rf_pred == 1, "fake", "real")
    out_df["rf_p_fake"] = rf_proba
    out_df["logreg_correct"] = out_df["label"] == out_df["logreg_pred"]
    out_df["rf_correct"] = out_df["label"] == out_df["rf_pred"]
    out_df.to_csv(predictions_csv, index=False)
    print(f"[{label}] Saved per-row LOOCV predictions to {predictions_csv}")

    misses = out_df[~out_df["logreg_correct"] | ~out_df["rf_correct"]]
    if len(misses) > 0:
        print(f"\n[{label}] {len(misses)} row(s) where at least one model was wrong:")
        print(misses[["filename", "label", "logreg_pred", "rf_pred"]].to_string(index=False))
    else:
        print(f"\n[{label}] Both models got every row right under LOOCV.")
        print(f"[{label}] At small n, treat a perfect score with suspicion, not celebration.")

    return True


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

    train_and_save(FEATURES_CSV, FEATURE_COLS, PREDICTIONS_CSV, MODEL_ARTIFACT, label="IMAGE")
    train_and_save(VIDEO_FEATURES_CSV, VIDEO_FEATURE_COLS, VIDEO_PREDICTIONS_CSV,
                    VIDEO_MODEL_ARTIFACT, label="VIDEO")


if __name__ == "__main__":
    main()
