# EVRA — Exoplanet Validation & Research with AI 🪐

**EVRA** (Exoplanet Validation & Research with AI) is a mission-aware **Mixture-of-Experts (MoE)** app for classifying exoplanet records using NASA’s **Kepler (KOI)**, **K2**, and **TESS** datasets.  
Built using Python Coding language and Streamlit library for a clean User-Interface ; you can **upload data → get predictions → evaluate metrics → retrain models → export** — with each expert preserving its original mission schema and feature engineering.

---

## 🔧 Quick Start

### 1) Install Requirements
```bash
pip install -r requirements.txt
```

## 2) Project Files

Place these files in your project folder:

### Models
*(or update paths via config.json)*

- `models/koi_lightgbm.pkl` (Kepler / KOI expert)
- `models/k2_lightgbm.pkl` (K2 expert)
- `models/tess_lightgbm.pkl` (TESS expert)

### Config
- `config.json` — points to model files (see example below)

**Example config.json:**

```json
{
  "models": {
    "kepler": "models/koi_lightgbm.pkl",
    "k2": "models/k2_lightgbm.pkl",
    "tess": "models/tess_lightgbm.pkl"
  }
}
```

### Assets
*(optional aesthetics)*

- `assets/bg.jpg` — dark exoplanets background

## 3) Run the App

To run the app, type the below command in a terminal or cmd (!! Make sure you have installed the requirements.txt in a virtual environment and activated it !!)

```bash
streamlit run app.py
```
Open in your browser: `http://localhost:8501`

---

## 🌌 What EVRA Does

**Classifies each record into one of:**
- **FALSE POSITIVE** • **CANDIDATE** • **CONFIRMED**

**Outputs class probabilities per row:**
- `p_FP` → probability of FALSE POSITIVE
- `p_PC` → probability of CANDIDATE  
- `p_CONF` → probability of CONFIRMED

### Routing Strategies *(choose in sidebar)*

- **Overlap (default)**: Automatically choose the single expert (Kepler/K2/TESS) whose feature overlap is highest after mission-specific preprocessing
- **Soft-vote**: Run all three experts and average their probabilities
- **Force a mission**: Manually select Kepler / K2 / TESS

### Candidate Thresholding *(optional)*
If the predicted top class is **CANDIDATE** but `max(prob) < threshold`, EVRA downgrades it to **FALSE POSITIVE**

---

## 🧭 UI Guide (Tabs)

### 1) 🚀 Upload & Predict
- Upload a CSV (any mission)
- EVRA routes to the best expert (Overlap/Soft-vote/Forced)
- Shows predictions + `p_FP`, `p_PC`, `p_CONF`
- Download predictions as CSV

### 2) 🧪 Predict Single Sample
- Enter one row using a few common numeric fields (missing values are NaN-safe for LightGBM)
- Get a single prediction + probabilities

### 3) 📈 Metrics & Stats
Upload a labeled CSV to evaluate performance:
- **Confusion matrix** (table)
- **Classification report** with Accuracy (%), Macro-F1, Weighted-F1
- **ROC-AUC** (macro OVR) when possible

**Auto-detect label columns:**
- **Kepler**: `koi_disposition`, `Disposition Using Kepler Data`
- **K2**: `disposition`, `Archive Disposition`, `archive_disposition`  
- **TESS**: `tfopwg_disp`, `TFOPWG Disposition`, `TFOPWG_DISP`, `disposition`

*Label normalization: maps variants to FALSE POSITIVE / CANDIDATE / CONFIRMED*

### 4) 🔍 Explainability
- Shows LightGBM feature importances for the currently selected expert
- Reflects the exact mission schema (including engineered columns)

### 5) 🧬 Ingest & Retrain
- Choose a mission to retrain: Kepler / K2 / TESS
- Upload your CSV (labeled recommended)
- EVRA applies that mission's preprocessing + FE, aligns to the expert's feature_names, splits 80/20 (stratified), trains LightGBM using sidebar hyperparameters, and shows validation metrics

**Pseudo-labeling (optional):**
- If your CSV lacks labels, EVRA can use its own high-confidence predictions (≥ your threshold) as temporary labels for retraining
- These pseudo-labels exist only in session; your CSV isn't modified
- *Always validate later with real labeled data*
- Export the retrained model as a `.pkl` directly from this tab

### 6) 🧰 Model Management
- Shows resolved paths for all three experts (from config.json)
- Export all current experts as one ZIP with their original filenames

**To make a retrained model permanent:**
1. Download the `.pkl` from Ingest & Retrain (or the ZIP here)
2. Move it to your models folder
3. Update the paths in `config.json`
4. Restart the app

### 7) ℹ️ About
Full explanation of routing, preprocessing, metrics, probabilities, retraining, and how EVRA uses NASA's free and open data

---

## 🎛️ Sidebar Controls

**Mission:** auto (router) or force kepler / k2 / tess

**Routing Strategy:** overlap (default) or softvote

**Candidate Threshold:** downgrade low-confidence CANDIDATE → FALSE POSITIVE

**Retraining Hyperparameters:**
- `learning_rate`
- `n_estimators` 
- `num_leaves`
- `subsample`
- `colsample_bytree`
- `min_child_samples`
- `reg_alpha`
- `reg_lambda`
- `class_weight`

---

## 🧬 Mission-Specific Preprocessing 
*(used for both Inference & Retraining)*

EVRA re-applies the exact training-time preprocessing for each expert during prediction and retraining, ensuring the feature layout matches what each model expects.

### Kepler (KOI) dataset

**Drop columns:**
`kepid`, `kepoi_name`, `kepler_name`, `koi_tce_delivname`, `koi_pdisposition`, `koi_vet_date`, `koi_vet_stat`, `koi_vet_url`

**Engineered features:**
- `log_prad = log1p(koi_prad)`
- `log_depth = log1p(koi_depth)`
- `duration_period_ratio = koi_duration / koi_period`
- `prad_srad_ratio = koi_prad / koi_srad`

### K2 dataset

**Drop columns (no extra FE):**
`pl_name`, `hostname`, `disp_refname`, `discoverymethod`, `disc_facility`, `soltype`, `pl_refname`, `st_refname`, `sy_refname`, `rastr`, `decstr`, `rowupdate`, `pl_pubdate`, `releasedate`, `default_flag`, `pl_bmassprov`

### TESS dataset

**Drop columns:**
`toi`, `tid`, `rastr`, `decstr`, `rowupdate`, `toi_created`

**Engineered features:**

**Logs:**
- `log_pl_orbper`
- `log_pl_trandep`
- `log_pl_rade`
- `log_pl_insol`
- `log_st_dist`
- `log_st_teff`
- `log_pl_eqt`

**Ratios / proxies:**
- `dur_per_ratio = pl_trandurh / pl_orbper`
- `rade_over_st_rad = pl_rade / st_rad`
- `pm_total = sqrt(st_pmra^2 + st_pmdec^2)`
- `insol_eqt_ratio = log1p(pl_insol) / log1p(pl_eqt)`
- `star_density_proxy ≈ st_mass / (st_rad^3)`
- `depth_flux_scaled = pl_trandep * 10^(-0.4 * st_tmag)` (SNR-ish proxy)

**Relative uncertainty features:**
- `relerr_pl_orbper`
- `relerr_pl_trandurh`
- `relerr_pl_trandep`
- `relerr_pl_rade`
- `relerr_st_teff`
- `relerr_st_dist`
- `relerr_st_tmag`

---

## 📂 Output Columns (Predictions)

When using **Upload & Predict**, EVRA appends these columns:

- `prediction` — final class label (FALSE POSITIVE / CANDIDATE / CONFIRMED)
- `p_FP` — probability of FALSE POSITIVE
- `p_PC` — probability of CANDIDATE
- `p_CONF` — probability of CONFIRMED

## ✅ Tips for Best Results

- Ensure numeric columns are truly numeric (avoid "—", "N/A", etc.)
- Provide mission-consistent fields (Kepler/K2/TESS) where possible
- If labels are imbalanced, try `class_weight='balanced'` before retraining
- For pseudo-labeling, use a high threshold (e.g., 0.90–0.95) to avoid reinforcing errors

## 🙌 Credits

Built by **Eva Ekhteyary** & **Rajdeep Roshan Sahu** for NASA Space Apps Challenge 2025.

Powered by NASA's free and open data from the NASA Exoplanet Archive (Kepler, K2, TESS).