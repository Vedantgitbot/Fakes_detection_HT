"""
features.py — Step 1: Feature Extraction

For every image in data/real_images/ and data/fake_images/, computes:
  - EXIF metadata presence/consistency
  - FFT frequency-domain artifact score
  - Error Level Analysis (ELA) variance
  - Noise-variance "perfection score"

Also supports video input: extracts N evenly-spaced frames, runs the same
four per-frame functions, and adds temporal-consistency features (variance
of each signal across frames).

Writes one row per image to outputs/features.csv
Writes one row per video to outputs/video_features.csv

Run from project root:
    python3 src/features.py
"""

import os
import io
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageFilter
import pandas as pd

# Pillow renamed resampling constants across versions:
#   Pillow >= 9.1: Image.Resampling.NEAREST
#   Pillow <  9.1: Image.NEAREST
NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")

# ---- paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(PROJECT_ROOT, "real_vs_fake", "real-vs-fake", "train", "real")
FAKE_DIR = os.path.join(PROJECT_ROOT, "real_vs_fake", "real-vs-fake", "train", "fake")
REAL_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "real_videos")
FAKE_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "fake_videos")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_CSV = os.path.join(OUT_DIR, "features.csv")
OUT_VIDEO_CSV = os.path.join(OUT_DIR, "video_features.csv")

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")
VALID_VIDEO_EXT = (".mp4", ".mov", ".avi", ".webm")


# ---------------- Feature 1: EXIF ----------------
def metadata_suspicion_flags(img: Image.Image, filepath: str):
    """
    Phase-1, no-pixel-analysis checks: dimensions + file-size-to-resolution
    ratio. KYC framing: a false positive here costs a legitimate user a
    failed verification, so this is a cheap PRE-FILTER/suspicion score, not
    a standalone verdict -- always pair with pixel-level signals before
    flagging a real submission.
    """
    w, h = img.size
    file_size = os.path.getsize(filepath)
    bytes_per_pixel = file_size / (w * h + 1e-8)
    # common StyleGAN/generative output sizes -- not proof, just a prior
    suspicious_dims = (w, h) in [(256, 256), (512, 512), (1024, 1024)]
    return {
        "width": w,
        "height": h,
        "bytes_per_pixel": bytes_per_pixel,
        "suspicious_dims": int(suspicious_dims),
    }


def check_exif(img: Image.Image):
    """Returns (has_exif: bool, has_camera_tag: bool)"""
    try:
        exif = img.getexif()
        if not exif or len(exif) == 0:
            return False, False
        # Tag 271 = Make, Tag 272 = Model (standard EXIF tags)
        has_camera_tag = 271 in exif or 272 in exif
        return True, has_camera_tag
    except Exception:
        return False, False


# ---------------- Feature 2: FFT high-frequency ratio ----------------
def fft_hf_ratio(img: Image.Image):
    """
    Converts to grayscale, computes 2D FFT, and returns the ratio of
    energy in the outer (high-frequency) annulus to total energy.
    Higher = more high-frequency artifact energy (common in generative upsampling).
    """
    # NEAREST, not bicubic -- bicubic resize is a low-pass filter and would
    # suppress the exact high-frequency content this feature measures.
    gray_img = img.convert("L").resize((256, 256), NEAREST)
    gray = np.array(gray_img, dtype=np.float64)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)

    # high-frequency = outer 30% radius band
    hf_mask = dist > (0.7 * max_dist)
    total_energy = magnitude.sum() + 1e-8
    hf_energy = magnitude[hf_mask].sum()

    return float(hf_energy / total_energy)


# ---------------- Feature 3: ELA variance ----------------
def ela_variance(img: Image.Image, quality: int = 90):
    """
    Re-saves image at a known JPEG quality, diffs against the (converted)
    original, and returns variance + mean of the difference map.

    KNOWN LIMITATION: sensitive to ORIGINAL compression history, not just
    authenticity -- a real photo saved at low quality (Q70) can show a very
    different signature than one saved at high quality (Q95), independent
    of whether it's fake. Check against actual feature distributions before
    trusting this as a strong standalone signal.
    """
    rgb = img.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(rgb, resaved)
    diff_arr = np.array(diff, dtype=np.float64)

    return float(diff_arr.var()), float(diff_arr.mean())


# ---------------- Feature 4: Noise-perfection score ----------------
def noise_perfection_score(img: Image.Image, patch_size: int = 16):
    """
    Extracts a high-frequency residual (image minus blurred version),
    splits into patches, and computes the variance-of-local-variances.
    Real sensor noise varies patch-to-patch (shadows vs highlights, etc).
    A LOW variance-of-variances means the noise is suspiciously uniform
    ("too perfect") -- often seen in generative outputs.
    """
    gray_img = img.convert("L").resize((256, 256))
    gray = np.array(gray_img, dtype=np.float64)

    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    blurred = np.array(Image.fromarray(gray_uint8).filter(
        ImageFilter.GaussianBlur(radius=2)
    ), dtype=np.float64)
    residual = gray - blurred

    h, w = residual.shape
    local_vars = []
    # range() intentionally drops any partial patch at bottom/right edge --
    # partial patches have artificially different variance stats than full ones.
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = residual[y:y + patch_size, x:x + patch_size]
            local_vars.append(patch.var())

    local_vars = np.array(local_vars)
    return float(local_vars.var())


# ---------------- Single-image feature bundle (shared by image + per-frame video path) ----------------
def extract_single_image_features(img: Image.Image, filepath: str = None) -> dict:
    has_exif, has_camera_tag = check_exif(img)
    fft_score = fft_hf_ratio(img)
    ela_var, ela_mean = ela_variance(img)
    noise_score = noise_perfection_score(img)
    feats = {
        "has_exif": int(has_exif),
        "has_camera_tag": int(has_camera_tag),
        "fft_hf_ratio": fft_score,
        "ela_variance": ela_var,
        "ela_mean": ela_mean,
        "noise_perfection_score": noise_score,
    }
    if filepath:
        feats.update(metadata_suspicion_flags(img, filepath))
    return feats


# ---------------- Video: frame extraction ----------------
def extract_frames(video_path: str, n_frames: int = 8):
    """
    Pulls n_frames evenly-spaced frames from a video via OpenCV, returns
    list of PIL Images (RGB). Small n_frames by design -- proof-of-concept,
    not full temporal coverage; running all four per-frame features is not
    cheap.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count from {video_path}")

    # evenly spaced frame indices across the video, avoiding first/last frame
    # (often black/transition frames in web-sourced clips)
    idxs = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


# ---------------- Video: temporal-consistency features ----------------
def video_temporal_features(video_path: str, n_frames: int = 8) -> dict:
    """
    Extracts frames, runs the four static features per-frame, then computes
    mean + variance of each signal ACROSS frames as temporal-consistency
    features. Real video: natural, small frame-to-frame drift. Reenacted /
    swapped video: either irregular spikes (identity artifacts popping in)
    or unnaturally flat consistency in a signal that should vary with real
    camera motion -- direction of the effect is empirical, not assumed.
    """
    frames = extract_frames(video_path, n_frames=n_frames)
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    per_frame = [extract_single_image_features(f) for f in frames]
    per_frame_df = pd.DataFrame(per_frame)

    signal_cols = ["fft_hf_ratio", "ela_variance", "ela_mean", "noise_perfection_score"]

    result = {"n_frames_used": len(frames)}
    for col in signal_cols:
        result[f"{col}_mean"] = float(per_frame_df[col].mean())
        # frame-to-frame consistency signal -- the new feature this section adds
        result[f"{col}_temporal_var"] = float(per_frame_df[col].var())

    return result


# ---------------- Main extraction loop: images ----------------
def extract_features_for_dir(directory: str, label: str):
    rows = []
    if not os.path.isdir(directory):
        print(f"  [!] Directory not found, skipping: {directory}")
        return rows

    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(VALID_EXT)])
    print(f"  Found {len(files)} images in {directory}")

    for fname in files:
        fpath = os.path.join(directory, fname)
        try:
            img = Image.open(fpath)
            img.load()
            feats = extract_single_image_features(img, filepath=fpath)
            feats["filename"] = fname
            feats["label"] = label
            rows.append(feats)
            print(f"    ✓ {fname}")
        except Exception as e:
            print(f"    ✗ {fname} -- failed: {e}")

    return rows


# ---------------- Main extraction loop: videos ----------------
def extract_features_for_video_dir(directory: str, label: str, n_frames: int = 8):
    rows = []
    if not os.path.isdir(directory):
        print(f"  [!] Directory not found, skipping: {directory}")
        return rows

    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(VALID_VIDEO_EXT)])
    print(f"  Found {len(files)} videos in {directory}")

    for fname in files:
        fpath = os.path.join(directory, fname)
        try:
            feats = video_temporal_features(fpath, n_frames=n_frames)
            feats["filename"] = fname
            feats["label"] = label
            rows.append(feats)
            print(f"    ✓ {fname} ({feats['n_frames_used']} frames)")
        except Exception as e:
            print(f"    ✗ {fname} -- failed: {e}")

    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Extracting features from REAL images...")
    real_rows = extract_features_for_dir(REAL_DIR, "real")

    print("Extracting features from FAKE images...")
    fake_rows = extract_features_for_dir(FAKE_DIR, "fake")

    all_rows = real_rows + fake_rows
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nWrote {len(df)} rows to {OUT_CSV}")
        print(df.describe(include="all"))
    else:
        print("\nNo images processed. Add images to data/real_images/ and data/fake_images/ first.")

    # video path is optional -- only runs if those dirs exist and have files
    print("\nExtracting features from REAL videos...")
    real_video_rows = extract_features_for_video_dir(REAL_VIDEO_DIR, "real")

    print("Extracting features from FAKE videos...")
    fake_video_rows = extract_features_for_video_dir(FAKE_VIDEO_DIR, "fake")

    all_video_rows = real_video_rows + fake_video_rows
    if all_video_rows:
        vdf = pd.DataFrame(all_video_rows)
        vdf.to_csv(OUT_VIDEO_CSV, index=False)
        print(f"\nWrote {len(vdf)} rows to {OUT_VIDEO_CSV}")
        print(vdf.describe(include="all"))
    else:
        print("\nNo videos processed (data/real_videos/, data/fake_videos/ empty or missing -- optional).")


if __name__ == "__main__":
    main()