# Fakes_detection_HT — Plurall AI Hackathon Build

## Fake Image/Video Triage Pipeline

A layered, honest proof-of-concept combining classical image-forensics, metadata pre-filtering, a pretrained CNN signal, video temporal-consistency, and an LLM reasoning layer to flag possibly AI-generated media.

**Disclaimer:** Portfolio/learning project, not a production-validated detection tool. Every result comes with an explicit confidence level and caveat.

## How It Works — Layered Pipeline

**Phase 1 — Metadata pre-filter (no pixel analysis):**
- Image dimensions, file-size-to-resolution ratio (`bytes_per_pixel`)
- Flag for common generative-model output dimensions (e.g. 256x256, 512x512, 1024x1024)
- EXIF presence / camera-tag presence
- Cheap, fast signals used as a soft pre-filter, never a standalone verdict — a false positive here costs a legitimate user a failed KYC verification

**Phase 2 — Pixel-level forensics:**
- FFT high-frequency energy ratio (GAN upsampling periodicity)
- Error Level Analysis (ELA) variance and mean (JPEG recompression inconsistency)
- Noise-residual "perfection" score (variance-of-local-variances — absence of natural sensor noise)
- **Pretrained signal:** VGG16-based classifier (`batch_predict.py`, external pretrained checkpoint) run as an additional, independent signal alongside the hand-engineered features — not a replacement

**Video support:**
- Extracts N evenly-spaced frames, runs the same four per-frame forensic functions, adds frame-to-frame temporal-consistency features (mean + variance of each signal across frames)

**Classification:**
- Logistic Regression & shallow Random Forest, evaluated with Leave-One-Out Cross-Validation (LOOCV) — appropriate given a small (150-image) training set
- Combined feature vector: 4 forensic signals + 4 metadata signals + 1 pretrained-model probability = 9 features total

**LLM Explanation:**
- Numeric features and dataset baselines (never the raw image/frames) passed to Groq-hosted Llama 3.3 70B for plain-language explanation
- Hard-constrained: never describes or invents visual detail, since it was never shown the image

**Post-training / improvement loop (design, not yet implemented):**
- Human review of low-confidence or classifier-disagreement predictions
- Confirmed labels fed back into the training set, prioritizing disagreements over confidently-correct cases
- Same human-in-the-loop principle as this author's separate Almaria project (AI-suggested fixes require human QC approval before entering the knowledge base)

## Dataset Used

Sourced from Plurall AI's hackathon challenge materials; a 150-image labeled subset (`real`/`fake`) sampled from the public **140k Real and Fake Faces** dataset (FFHQ vs. StyleGAN, Kaggle) for tractable iteration under time constraints — same small-n reasoning behind the LOOCV choice.

50-image unlabeled holdout set scored for final deliverable, with self-estimated precision/recall (no ground truth available for holdout).

## Quick Start

```bash
pip install -r requirements.txt
# .env: GROQ_API_KEY=your_key

python3 src/features.py              # forensic + metadata features -> outputs/features.csv
python3 batch_predict.py <folder> <output.csv>   # pretrained VGG16 signal, run separately
python3 src/model.py                 # merges VGG signal in, trains + evaluates, saves artifacts
streamlit run dashboard/main.py      # local dashboard
```

## Project Structure

```
├── ai/         # LLM reasoning layer (Groq API, prompt, structured output)
├── dashboard/  # Streamlit UI
├── src/        # features.py (extraction), model.py (training/eval)
├── data/       # real_images/, fake_images/ (sampled subset, not committed)
└── outputs/    # features.csv, vgg_predictions_merged.csv, model artifacts (not committed)
```

## Honest Design Choices & Limitations

**What's deliberate:**
- LLM structurally forbidden from describing the image — numeric signals only
- Bug caught and fixed: bicubic resize was silently low-pass-filtering the FFT high-frequency signal before extraction
- Metadata checks framed explicitly around false-positive cost in a KYC context, not treated as a standalone verdict
- LOOCV instead of train/test split, given small n; a perfect score is treated as a red flag, not a win
- VGG16 signal kept separate and additive, not blended silently — visible as its own column for scrutiny

**Known limitations:**
- 150-image training sample is a directional estimate, not a generalized benchmark
- FFT/noise signals are tuned toward GAN-specific artifacts; not validated against diffusion-model-generated fakes
- ELA remains confounded by original compression history, not just authenticity
- Video temporal-variance features have no established real-vs-fake baseline direction yet — exploratory
- Local, single-user scope — no API, auth, or production hardening
- Holdout precision/recall is self-estimated without ground truth, not independently verified

## Stack

Python • Pillow • NumPy • pandas • scikit-learn • joblib • Streamlit • Groq API (Llama 3.3 70B) • PyTorch/torchvision (VGG16) • OpenCV (video frame extraction)
