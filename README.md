# Chatter Detection in Turning Operations using Supervised Machine Learning

A physics-informed approach to detecting regenerative chatter — a self-excited tool vibration — from raw cutting-force sensor data collected during single-point turning.

**Course Project — ME623, Dynamics of Machining Processes, Dept. of Mechanical Engineering, IIT Guwahati**

---

## Overview

Chatter is a self-excited vibration between the cutting tool and workpiece that degrades surface finish, accelerates tool wear, and can damage the tool entirely. This project builds a complete pipeline to detect it automatically from 3-axis cutting-force signals, with no expert-labeled ground truth available — the core challenge, and the main technical contribution of this project, is building a **defensible label** from unlabeled sensor data before any model can be trained at all.

## Dataset

- 18 single-point turning experiments, 3-component piezoelectric dynamometer, **10 kHz** sampling rate
- Measures feed force (Fx), radial force (Fy), and cutting/tangential force (Fz)
- Spans 3 spindle speeds (250 / 325 / 420 RPM) and depths of cut from 100–1400 µm

## Methodology

### 1. Engagement detection
Every raw recording includes several seconds of the spindle running before the tool actually touches the workpiece (and again after it disengages). An automatic, per-file detector isolates the true cutting window using resultant-force RMS thresholding, with a secondary check that guards against a piezoelectric charge-drift sensor artifact found in the heaviest cut.

### 2. Feature extraction
1-second sliding windows (50% overlap) over the engaged region only, extracting time-domain statistics (mean, std, RMS, skewness, kurtosis, energy, etc.), frequency-domain statistics (FFT peak amplitude, dominant frequency), and dimensionless condition-monitoring indices, per axis.

### 3. A physics-informed chatter label
With no expert labels available, chatter is identified from first principles rather than an arbitrary amplitude cutoff:
- **Amplitude ratio** — the dominant FFT peak in each window is compared against the strongest expected forced-vibration harmonic (an integer multiple of the spindle rotation frequency) in that same window
- **Harmonic exclusion** — a peak that *is* a spindle-rotation harmonic is forced vibration, not chatter, regardless of amplitude
- **Persistence filtering** — the elevated, non-harmonic peak must sustain across consecutive overlapping windows, reflecting chatter's self-excited, growing character rather than a one-off transient

This label is validated against **stability-lobe theory**: chatter probability rises sharply and consistently once depth of cut exceeds a critical value — a pattern the label reproduces without ever being told the cutting condition, using only each window's own force spectrum.

### 4. Modeling
- Feature selection via correlation, GroupKFold-averaged Random Forest importance, and SelectKBest
- A deliberate check for label leakage: engineered "peak-concentration" features found to conceptually overlap with the label's own construction were identified and removed before modeling
- 8 classifiers benchmarked (Random Forest, XGBoost, SVM, Logistic Regression, and balanced/ensemble variants) using **GroupKFold cross-validation** (grouped by cutting trial)
- Final model selected by **F2-score**, weighting recall over precision to reflect the real cost asymmetry of missing a chatter event

## Results

| Metric | Score |
|---|---|
| Precision | 0.89 |
| Recall | 0.97 |
| F1-score | 0.93 |
| F2-score | 0.95 |

The final model (Random Forest) is used to generate a **stability diagram** — predicted chatter probability across every combination of depth of cut and spindle speed — for direct use in process planning.

## Repository Contents

| File | Description |
|---|---|
| `ME623_Chatter_Detection.ipynb` | Full, executed analysis notebook — data cleaning through final model |
| `final_all_features_dataset.csv` | Engineered feature dataset (739 windows, 17 trials), including both the naive and physics-informed labels |
| `ME623_project_report.docx` | Full written report: methodology, literature review, results, limitations |
| `ME623_Chatter_Detection_Deck.pdf` | Presentation deck |

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `SciPy (FFT)` · `Matplotlib` / `Seaborn`

## Key Learnings

- Built an automatic engagement-detection method for raw sensor recordings and validated it was necessary via a controlled ablation study
- Designed a novel labeling technique from unlabeled signal data using frequency-domain physics rather than an arbitrary statistical cutoff
- Diagnosed and corrected a subtle feature/label overlap issue before trusting model results — a construct-validity check that mattered more than the headline accuracy number
- Selected a model using a cost-sensitive metric (F2) rather than defaulting to accuracy or F1

## Limitations

The label, while physically grounded, is still derived from the same signal used to build the model's features — it has not been validated against an independent measurement (accelerometer, microphone, or post-process surface roughness). This is the clearest next step for extending the work; see the full report for a complete discussion.

---

*Group 7 — Saket Mehla (230103087), Raghav Kapoor (230103081) — IIT Guwahati*
