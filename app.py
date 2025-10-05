# app.py — EVRA: Exoplanet Detector
# ---------------------------------------------------------------
# Features:
# - Mixture-of-Experts (KOI/K2/TESS) with routing:
#     • overlap (default): pick expert with most matching features
#     • soft-vote: average probabilities across all experts
# - Upload CSVs or enter a single sample
# - Metrics tab (classification report, confusion matrix, ROC-AUC when labels are present)
# - Ingest & retrain on uploaded data (manual button; hyperparameters set in sidebar)
# - Mission-specific preprocessing & feature engineering (KOI/K2/TESS)
# - Explainability (LightGBM feature importances)
# - Download predictions and export retrained models
# - Export all current experts as a ZIP (Model Management)
#
# Notes:
# - Candidate thresholding (optional): low-confidence CANDIDATE can be downgraded to FALSE POSITIVE
# - Pseudo-labeling toggle exists in UI but is not currently applied in retraining code
#
# Requirements:
#   pip install -r requirements.txt
#
# Model paths:
#   Set in config.json under:
#     {
#       "models": {
#         "kepler": "path/to/koi_model.pkl",
#         "k2":     "path/to/k2_model.pkl",
#         "tess":   "path/to/tess_model.pkl"
#       }
#     }
#
# Tip:
# - Each expert expects its own mission-specific features; EVRA re-applies the same preprocessing used at training time.

import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import zipfile
from base64 import b64encode


# -------------------- Theming & Layout --------------------
st.set_page_config(
    page_title="EVRA • NASA Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
)

st.markdown("""
<style>
:root{
  --bg:#070B1A;        /* deep space */
  --panel:#0E1633;     /* darker panel */
  --accent:#E74C3C;    /* NASA red */
  --accent2:#4FC3F7;   /* cerulean */
  --text:#E6EDF7;
  --muted:#98A2B3;
}
html, body {
  color: var(--text);
  background: radial-gradient(1200px 800px at 10% -10%, #162047 0%, #0B1127 40%, #070B1A 100%) !important;
}
.block-container { padding-top: 0.8rem; }

h1, h2, h3 { color: var(--text); }
small, .stCaption, .stText, .st-emotion-cache { color: var(--muted); }

.sidebar .sidebar-content, section[data-testid="stSidebar"] > div {
  background: linear-gradient(180deg, #0E1633 0%, #0A0F22 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stHeader"] {
  background: transparent;
}
.stButton>button, .stDownloadButton>button {
  background: linear-gradient(90deg, var(--accent) 0%, #FF7B6E 100%);
  color: white; border: 0; border-radius: 10px; padding: 0.6rem 1rem;
  transition: transform .15s ease-in-out, box-shadow .15s;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  transform: translateY(-1px) scale(1.01);
  box-shadow: 0 6px 20px rgba(231,76,60,0.3);
}
a, .stMarkdown a { color: var(--accent2) !important; text-decoration: none; }

.card {
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px; padding: 1rem; margin-bottom: 1rem;
}
.badge {
  display: inline-block; padding: .2rem .5rem; border-radius: 999px;
  font-size: .8rem; background: rgba(79,195,247,0.15); color: var(--accent2);
  border: 1px solid rgba(79,195,247,0.3);
}
.title-anim {
  background: linear-gradient(90deg, #FFFFFF 0%, #A6C8FF 40%, #4FC3F7 60%, #FFFFFF 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shine 6s linear infinite;
  background-size: 200% auto;
}
@keyframes shine { to { background-position: 200% center; } }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent); }
</style>
""", unsafe_allow_html=True)

def _b64(path: Path) -> str:
    with open(path, "rb") as f:
        return b64encode(f.read()).decode("utf-8")

try:
    bg_path = Path(__file__).parent / "assets" / "bg.jpg"       #   background image
    bg_b64 = _b64(bg_path)

    st.markdown(
        f"""
        <style>
        /* Make app surfaces transparent so the background layer shows through */
        html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
            background: transparent !important;
        }}

        /* Single background layer attached to the app view container */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: url("data:image/jpg;base64,{bg_b64}") center/cover no-repeat fixed;
            opacity: 0.08;                /* adjust 0.05–0.12 for subtlety */
            pointer-events: none;         /* don't block clicks */
            z-index: -1;                  /* behind all app content */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.warning(f"Background image not found or failed to load: {e}")
    
st.markdown(
    f"""
    <style>
    /* Ensure the app container participates in stacking and spans the page */
    [data-testid="stAppViewContainer"] {{
        position: relative;
        min-height: 100vh;
        background: transparent !important;
    }}

    /* Semi-transparent background image layer (no negative z-index) */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: url("data:image/jpeg;base64,{bg_b64}") center/cover no-repeat fixed;
        opacity: 0.30;                 /* tweak 0.06–0.12 for subtlety */
        pointer-events: none;           /* don't block clicks */
        z-index: 0;                     /* keep it behind content but visible */
    }}

    /* Keep all visible surfaces above the bg */
    .main, .block-container, [data-testid="stSidebar"], header, footer {{
        position: relative;
        z-index: 1;
        background: transparent !important;
    }}

    /* (Optional) Remove any leftover default backgrounds that might obscure */
    html, body {{
        background: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
/* Force the sidebar to be scrollable */
section[data-testid="stSidebar"] > div {
  height: 100vh;                /* full viewport height */
  max-height: 100vh;            /* avoid exceeding viewport */
  overflow-y: auto !important;  /* enable vertical scroll */
  overscroll-behavior: contain; /* keep scroll within sidebar */
  padding-bottom: 1rem;         /* small bottom spacing */
}

/* Backup selector for newer Streamlit builds */
[data-testid="stSidebarContent"] {
  height: 100vh;
  overflow-y: auto !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Constants & Helpers --------------------


class IdentityImputer:
    """Pass-through imputer for inference when the model can handle NaNs (LightGBM)."""
    def fit(self, X, y=None): return self
    def transform(self, X): return X

COMMON_ORDER = np.array(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"], dtype=object)

def normalize_bundle(bundle):
    """
    Return a normalized 4-tuple: (model, label_encoder, feature_names, imputer)
    Accepts: dict; 4-tuple; 3-tuple; raw LGBMClassifier.
    """
    # dict save
    if isinstance(bundle, dict):
        model = bundle.get("model", None)
        le = bundle.get("label_encoder", None)
        feat_names = bundle.get("feature_names", None)
        imputer = bundle.get("imputer", None)

    # tuple saves
    elif isinstance(bundle, (list, tuple)):
        if len(bundle) == 4:
            model, le, feat_names, imputer = bundle
        elif len(bundle) == 3:
            model, le, feat_names = bundle
            imputer = None
        else:
            raise ValueError(f"Unsupported bundle tuple length: {len(bundle)}")

    # raw model (e.g., only LightGBM was saved)
    else:
        model = bundle
        le = None
        feat_names = getattr(model, "feature_name_", None)
        imputer = None

    # Defaults/fallbacks
    if le is None:
        le = LabelEncoder().fit(COMMON_ORDER)
    if feat_names is None:
        raise ValueError("feature_names are missing and could not be inferred from the model.")
    if imputer is None:
        # For inference, LightGBM can handle NaNs; so a no-op imputer is fine.
        imputer = IdentityImputer()

    return (model, le, feat_names, imputer)

def serialize_bundle_for_disk(bundle):
    """
    Prepare a model bundle for joblib/pickle.
    Drops IdentityImputer (set to None) because custom classes in __main__
    are fragile across Streamlit reruns.
    """
    model, le, feat_names, imputer = normalize_bundle(bundle)
    return {
        "model": model,
        "label_encoder": le,
        "feature_names": list(feat_names),
        "imputer": None
    }

@st.cache_resource(show_spinner=False)
def load_model_paths(cfg_path: str = "config.json"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    def norm(p): return str(Path(p).expanduser().resolve())
    koi_path  = norm(cfg["models"]["kepler"])
    k2_path   = norm(cfg["models"]["k2"])
    tess_path = norm(cfg["models"]["tess"])

    # sanity check
    for name, p in [("KOI", koi_path), ("K2", k2_path), ("TESS", tess_path)]:
        if not Path(p).exists():
            st.error(f"{name} model not found at: {p}")
            st.stop()
    return koi_path, k2_path, tess_path

def save_model_paths_to_config(koi_path: str, k2_path: str, tess_path: str, cfg_path: str = "config.json"):
    payload = {"models": {"kepler": koi_path, "k2": k2_path, "tess": tess_path}}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def norm_label_generic(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper()
    if s in {"CONFIRMED", "CONFIRMED PLANET", "KP", "KNOWN PLANET"}: return "CONFIRMED"
    if s in {"CANDIDATE", "PLANET CANDIDATE", "PC", "APC", "AMBIGUOUS PLANETARY CANDIDATE"}: return "CANDIDATE"
    if s in {"FALSE POSITIVE", "FP"}: return "FALSE POSITIVE"
    return np.nan

def autodetect_target_col(df, mission_hint=None):
    if mission_hint == "kepler":
        for c in ["koi_disposition", "Disposition Using Kepler Data"]: 
            if c in df.columns: return c
    if mission_hint == "k2":
        for c in ["disposition", "Archive Disposition", "archive_disposition"]:
            if c in df.columns: return c
    if mission_hint == "tess":
        for c in ["tfopwg_disp", "TFOPWG Disposition", "TFOPWG_DISP", "disposition"]:
            if c in df.columns: return c
    # generic fallback
    for c in ["koi_disposition", "tfopwg_disp", "disposition", "Archive Disposition"]:
        if c in df.columns:
            return c
    return None

# ---------- Dataset-specific Feature Engineering (KOI, K2, TESS) ----------

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
        if isinstance(out, pd.Series):
            out[~np.isfinite(out)] = np.nan
        else:
            out = np.where(np.isfinite(out), out, np.nan)
        return out

def add_relerr(X, base, e1, e2):
    """Add relative uncertainty feature: relerr_base = 0.5*(|e1|+|e2|)/|base|"""
    if base in X.columns and e1 in X.columns and e2 in X.columns:
        denom = X[base].abs().replace(0, np.nan)
        rel = 0.5 * (X[e1].abs() + X[e2].abs()) / denom
        X[f"relerr_{base}"] = rel

def preprocess_k2(df: pd.DataFrame) -> pd.DataFrame:
    """Apply K2 training-time drops only (no FE beyond your original recipe)."""
    drop_cols = [
        "pl_name","hostname","disp_refname","discoverymethod","disc_facility","soltype",
        "pl_refname","st_refname","sy_refname","rastr","decstr","rowupdate",
        "pl_pubdate","releasedate","default_flag","pl_bmassprov"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

def preprocess_koi(df: pd.DataFrame) -> pd.DataFrame:
    """Apply KOI training-time drops and engineered features (exactly as used in training)."""
    drop_cols = [
        'kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname',
        'koi_pdisposition', 'koi_vet_date', 'koi_vet_stat', 'koi_vet_url'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Feature engineering
    X = df.copy()
    # Logs
    if "koi_prad" in X.columns:
        X['log_prad'] = np.log1p(X['koi_prad'].clip(lower=0))
    if "koi_depth" in X.columns:
        X['log_depth'] = np.log1p(X['koi_depth'].clip(lower=0))
    # Ratios
    if {"koi_duration", "koi_period"}.issubset(X.columns):
        X['duration_period_ratio'] = safe_div(X['koi_duration'], X['koi_period'])
    if {"koi_prad", "koi_srad"}.issubset(X.columns):
        X['prad_srad_ratio'] = safe_div(X['koi_prad'], X['koi_srad'])
    return X

def preprocess_tess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TESS training-time drops and engineered features (exactly as used in training)."""
    drop_cols = ["toi","tid","rastr","decstr","rowupdate","toi_created"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    X = df.copy()

    # 6a) Log transforms (tame skew)
    for col in ["pl_orbper","pl_trandep","pl_rade","pl_insol","st_dist","st_teff","pl_eqt"]:
        if col in X.columns:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))

    # 6b) Ratios / proxies
    if {"pl_trandurh","pl_orbper"}.issubset(X.columns):
        X["dur_per_ratio"] = safe_div(X["pl_trandurh"], X["pl_orbper"])

    if {"pl_rade","st_rad"}.issubset(X.columns):
        X["rade_over_st_rad"] = safe_div(X["pl_rade"], X["st_rad"])

    if {"st_pmra","st_pmdec"}.issubset(X.columns):
        X["pm_total"] = np.sqrt(X["st_pmra"]**2 + X["st_pmdec"]**2)

    if {"pl_insol","pl_eqt"}.issubset(X.columns):
        X["insol_eqt_ratio"] = safe_div(
            np.log1p(X["pl_insol"].clip(lower=0)),
            np.log1p(X["pl_eqt"].clip(lower=0))
        )

    # crude star-density proxy ~ M / R^3
    if {"st_mass","st_rad"}.issubset(X.columns):
        X["star_density_proxy"] = safe_div(X["st_mass"], (X["st_rad"]**3))

    # flux ~ 10^(-0.4 * mag); SNR-ish scaling
    if {"pl_trandep","st_tmag"}.issubset(X.columns):
        flux = np.power(10.0, -0.4 * X["st_tmag"].fillna(X["st_tmag"].median()))
        X["depth_flux_scaled"] = X["pl_trandep"].fillna(0) * flux

    # 6c) Uncertainty features (relative errors)
    add_relerr(X, "pl_orbper",   "pl_orbpererr1",   "pl_orbpererr2")
    add_relerr(X, "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2")
    add_relerr(X, "pl_trandep",  "pl_trandeperr1",  "pl_trandeperr2")
    add_relerr(X, "pl_rade",     "pl_radeerr1",     "pl_radeerr2")
    add_relerr(X, "st_teff",     "st_tefferr1",     "st_tefferr2")
    add_relerr(X, "st_dist",     "st_disterr1",     "st_disterr2")
    add_relerr(X, "st_tmag",     "st_tmagerr1",     "st_tmagerr2")

    return X

# -----------------------------------------------------------------------------

def _fmt_pct(x):
    try:
        if x is None or np.isnan(x):
            return "—"
        return f"{x * 100:.1f}%"
    except Exception:
        return "—"

def show_clean_classification_report(
    y_true_idx,
    y_pred_idx,
    class_names=COMMON_ORDER,
    proba=None,             
    key_prefix="rep",
    title=None             
):
    if title:
        st.markdown(f"#### {title}")

    rep_str = classification_report(
        y_true_idx, y_pred_idx,
        target_names=class_names,
        digits=3,
        zero_division=0
    )

    rep_dict = classification_report(
        y_true_idx, y_pred_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    try:
        acc = float(rep_dict.get("accuracy", (y_true_idx == y_pred_idx).mean()))
    except Exception:
        acc = float((y_true_idx == y_pred_idx).mean())

    macro_f1 = float(rep_dict.get("macro avg", {}).get("f1-score", float("nan")))
    weighted_f1 = float(rep_dict.get("weighted avg", {}).get("f1-score", float("nan")))

    # Summary metrics
    def _fmt_pct(x):
        try:
            if x is None or np.isnan(x):
                return "—"
            return f"{x * 100:.1f}%"
        except Exception:
            return "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _fmt_pct(acc))
    c2.metric("Macro F1", f"{macro_f1:.3f}")
    c3.metric("Weighted F1", f"{weighted_f1:.3f}")

    if proba is not None:
        try:
            roc = roc_auc_score(y_true_idx, proba, multi_class="ovr")
            st.caption(f"**ROC-AUC (macro OVR):** {roc:.4f}")
        except Exception:
            pass

    # report in nicely formatted block
    st.code(rep_str, language="text")

@st.cache_resource(show_spinner=False)
def load_bundle(path):
    raw = joblib.load(path)
    return normalize_bundle(raw)

def reindex_impute(df, feat_names, imputer):
    X = df.reindex(columns=feat_names, fill_value=np.nan)
    return pd.DataFrame(imputer.transform(X), columns=feat_names)

def proba_in_common(bundle, df):
    model, le, feat_names, imputer = normalize_bundle(bundle) 
    X = df.reindex(columns=feat_names, fill_value=np.nan)
    X = pd.DataFrame(imputer.transform(X), columns=feat_names)
    P = model.predict_proba(X)

    out = np.zeros((len(X), len(COMMON_ORDER)))
    for i, c in enumerate(le.classes_):
        out[:, np.where(COMMON_ORDER == c)[0][0]] = P[:, i]
    return out

def predict_router(df, source, koi, k2b, tess, strategy="softvote"):
    """
    Apply the *training-time* preprocessing for each expert before prediction,
    so feature names (incl. engineered ones) match what the model expects.
    """
    # Preprocess once per expert
    D_koi  = preprocess_koi(df.copy())
    D_k2   = preprocess_k2(df.copy())
    D_tess = preprocess_tess(df.copy())

    experts = {
        "kepler": (koi,  D_koi),
        "k2":     (k2b,  D_k2),
        "tess":   (tess, D_tess),
    }

    def overlap(d, feat_names):
        return sum(col in d.columns for col in feat_names)

    # If a source is forced, just use that expert + its preprocessed df
    if source in experts:
        bundle, dproc = experts[source]
        P = proba_in_common(bundle, dproc)
        return COMMON_ORDER[P.argmax(1)], P, source

    # Overlap routing (compare with *processed* dfs to include engineered cols)
    if strategy == "overlap":
        ovs = {
            "kepler": overlap(D_koi,  koi[2]),
            "k2":     overlap(D_k2,   k2b[2]),
            "tess":   overlap(D_tess, tess[2]),
        }
        best = max(ovs, key=ovs.get)
        bundle, dproc = experts[best]
        P = proba_in_common(bundle, dproc)
        return COMMON_ORDER[P.argmax(1)], P, f"overlap→{best}"

    # Soft-vote: average probs from all experts (each on its own preprocessed df)
    Ps = [
        proba_in_common(koi,  D_koi),
        proba_in_common(k2b,  D_k2),
        proba_in_common(tess, D_tess),
    ]
    Pavg = np.mean(Ps, axis=0)
    return COMMON_ORDER[Pavg.argmax(1)], Pavg, "soft-vote"

def class_index_map():
    enc = LabelEncoder().fit(COMMON_ORDER)
    return enc

# -------------------- Sidebar Controls --------------------
st.sidebar.title("🛰️ Controls")
st.sidebar.caption("Configure mission routing, strategy, training, and thresholds.")

# load defaults from config.json
koi_path, k2_path, tess_path = load_model_paths()

# koi_path  = st.sidebar.text_input("KOI model path",  koi_path)
# k2_path   = st.sidebar.text_input("K2 model path",   k2_path)
# tess_path = st.sidebar.text_input("TESS model path", tess_path)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Prediction Settings")

# Routing & strategy
colA, colB = st.sidebar.columns(2)
mission_options = ["auto", "kepler", "k2", "tess"]
if "mission_choice" not in st.session_state:
    st.session_state["mission_choice"] = "auto"

with colA:
    mission_choice = st.radio(
        "Mission",
        mission_options,
        index=mission_options.index(st.session_state["mission_choice"]),
        key="mission_choice",
        help=(
            "Select which expert to use:\n\n"
            "• auto — Router decides using your chosen strategy:\n"
            "   – softvote or overlap\n\n"
            "• kepler — Force Kepler expert\n"
            "• k2 — Force K2 expert\n"
            "• tess — Force TESS expert"
        ),
    )

# --- Strategy (stateful, default = overlap) ---
strategy_options = ["overlap", "softvote"]
if "strategy_choice" not in st.session_state:
    st.session_state["strategy_choice"] = "overlap"

with colB:
    strategy = st.radio(
        "Routing Strategy",
        strategy_options,
        index=strategy_options.index(st.session_state["strategy_choice"]),
        key="strategy_choice",
        help=(
            "overlap: Pick the single expert with the most matching features.\n\n"
            "softvote: Average probabilities of Kepler+K2+TESS and pick the top class."
        ),
    )

# Reactive mini-explainers 
def _mission_note(m):
    if m == "auto":
        return "Auto: the router follows your selected strategy (softvote or overlap)."
    return f"Forced mission: using only **{m.upper()}** expert for predictions."

def _strategy_note(s):
    return (
        "Strategy • Soft-vote: averages probabilities from all experts."
        if s == "softvote"
        else "Strategy • Overlap: picks the single expert with the most matching features."
    )

st.sidebar.caption(_mission_note(mission_choice))
st.sidebar.caption(_strategy_note(strategy))

# Thresholds & pseudo-labeling
threshold_candidate = st.sidebar.slider("Min prob for CANDIDATE (threshold)", 0.0, 1.0, 0.8, 0.01,help="Threshold value to classify the predictions. If the top class is CANDIDATE but its probability is below this threshold, it is downgraded to FALSE POSITIVE.")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Ingest & Retrain Settings")

pseudo_lab = st.sidebar.checkbox("Enable pseudo-labeling on upload (semi-supervised)",
                                value=False,
                                 help=(
                                    "Pseudo-labeling uses the model’s own high-confidence predictions as temporary labels for rows that "
                                    "lack ground-truth labels, so you can include them in retraining.\n\n"
                                    "How it works:\n"
                                    "• The model predicts class probabilities for each row.\n"
                                    "• Only rows with max probability ≥ the threshold (below) are used as pseudo-labeled samples.\n\n"
                                    "Notes:\n"
                                    "• Original CSV is not modified; pseudo-labels live only in memory for this session.\n"
                                    "• Use a high threshold (e.g., 0.90–0.95) to avoid reinforcing mistakes.\n"
                                    "• Helpful when you have few/no labels, but always validate on a truly labeled set."
                                )
                                )
pseudo_thresh = st.sidebar.slider("Pseudo-label min confidence", 0.5, 0.99, 0.9, 0.01) if pseudo_lab else 0.9

# Hyperparams for retraining
with st.sidebar.expander("⚙️ Hyperparameters (retrain)"):
    lr = st.number_input("learning_rate", 0.001, 0.5, 0.05, 0.01)
    nest = st.number_input("n_estimators", 100, 3000, 1000, 100)
    leaves = st.number_input("num_leaves", 15, 255, 63, 1)
    subs = st.slider("subsample", 0.5, 1.0, 0.9, 0.05)
    cols = st.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
    minchild = st.number_input("min_child_samples", 5, 200, 30, 1)
    rega = st.number_input("reg_alpha", 0.0, 5.0, 0.1, 0.1)
    regl = st.number_input("reg_lambda", 0.0, 10.0, 1.0, 0.1)
    balance = st.checkbox("class_weight='balanced'", value=False)

# -------------------- Load Models --------------------
try:
    koi  = load_bundle(koi_path)
    k2b  = load_bundle(k2_path)
    tess = load_bundle(tess_path)
    st.sidebar.success("Models loaded.")
except Exception as e:
    st.sidebar.error(f"Load error: {e}")
    st.stop()

# ---- In-session overrides (persist until reload) ----
if "expert_overrides" not in st.session_state:
    st.session_state["expert_overrides"] = {}
_ov = st.session_state["expert_overrides"]

# Apply overrides if present
koi  = _ov.get("kepler", koi)
k2b  = _ov.get("k2", k2b)
tess = _ov.get("tess", tess)

# small status hints in the sidebar
for _name, _bundle in [("KEPLER", koi), ("K2", k2b), ("TESS", tess)]:
    if _name.lower() in _ov:
        st.sidebar.caption(f"Using in-session override for **{_name}**")

# -------------------- Header --------------------
st.markdown(
    """
    <div style="padding:1.0rem 0 0.2rem 0;">
      <h1 class="title-anim">EVRA • Exoplanet Validation & Research with AI</h1>
      <div class="badge">Kepler</div> <div class="badge">K2</div> <div class="badge">TESS</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("🪐 Classify records into **FALSE POSITIVE**, **CANDIDATE**, or **CONFIRMED** using a Mixture-of-Experts trained on NASA datasets. Upload CSVs or enter a single sample, view metrics, re-train with your data, and export models.")

# -------------------- Tabs --------------------
tabs = st.tabs([
    "🚀 Upload & Predict",
    "🧪 Predict Single Sample",
    "📈 Metrics & Stats",
    "🔍 Explainability",
    "🧬 Ingest & Retrain",
    "🧰 Model Management",
    "ℹ️ About"
])

# ===== 1) Upload & Predict =====
with tabs[0]:
    st.markdown("### Upload CSV")
    st.caption("Upload Kepler/K2/TESS datasets for exoplanet detection.\n\n This page is for **Prediction only** - a label column (in uploaded dataset) is **optional** here and is **ignored** during prediction. If a label column exists, switch to **Metrics & Stats** to evaluate performance.")

    up = st.file_uploader("Upload CSV file", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Preview", df.head())
        st.markdown('</div>', unsafe_allow_html=True)

        use_source = None if mission_choice == "auto" else mission_choice
        labels, P, used = predict_router(df, use_source, koi, k2b, tess, strategy=strategy)

        # Optional candidate thresholding
        if threshold_candidate > 0.0:
            maxp = P.max(axis=1)
            argm = P.argmax(axis=1)
            for i in range(len(labels)):
                if labels[i] == "CANDIDATE" and maxp[i] < threshold_candidate:
                    labels[i] = "FALSE POSITIVE"

        out = df.copy()
        out["prediction"] = labels
        out["p_FP"]   = P[:, 0]
        out["p_PC"]   = P[:, 1]
        out["p_CONF"] = P[:, 2]

        st.success(f"Predicted {len(out)} rows • Router: **{used}**")
        st.dataframe(out.head(20), use_container_width=True)

        # Distribution chart
        st.markdown("#### Prediction distribution")
        st.bar_chart(out["prediction"].value_counts())

        # Download predictions
        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv"
        )

# ===== 2) Single Sample =====
with tabs[1]:
    st.markdown("### Single Sample")
    st.caption("Enter a few common fields. Missing values are imputed using the selected expert’s schema.")

    use_source = None if mission_choice == "auto" else mission_choice
    bundle = {"kepler": koi, "k2": k2b, "tess": tess}.get(use_source, tess)
    _, _, feat_names, _ = bundle
    row = {c: np.nan for c in feat_names}

    def put(col, label, default):
        cval = st.text_input(label, value=str(default))
        try:
            val = float(cval)
            if col in row: row[col] = val
        except:
            pass

    c1, c2, c3 = st.columns(3)
    with c1:
        put("pl_orbper", "Orbital period (days)", 5.1)
        put("pl_trandurh", "Transit duration (hours)", 3.0)
        put("pl_trandep", "Transit depth (ppm)", 800.0)
    with c2:
        put("pl_rade", "Planet radius (Rearth)", 2.1)
        put("pl_insol", "Insolation (Earth=1)", 500.0)
        put("pl_eqt", "Equilibrium temperature (K)", 1000.0)
    with c3:
        put("st_teff", "Star Teff (K)", 5700)
        put("st_rad", "Star radius (Rsun)", 1.0)
        put("st_mass", "Star mass (Msun)", 1.0)

    if st.button("Predict single sample"):
        df1 = pd.DataFrame([row])
        labels, P, used = predict_router(df1, use_source, koi, k2b, tess, strategy=strategy)
        # Optional threshold
        if threshold_candidate > 0.0 and labels[0] == "CANDIDATE" and P[0].max() < threshold_candidate:
            labels[0] = "FALSE POSITIVE"
        st.success(f"Router: **{used}**")
        st.markdown(f"**Prediction:** {labels[0]}")
        # Show percentages in mission order (FP, Candidate, Confirmed)
        cols = st.columns(3)
        for i, lab in enumerate(COMMON_ORDER):
            cols[i].metric(lab.title(), f"{P[0][i]*100:.2f}%")

# ===== 3) Metrics & Stats =====
with tabs[2]:
    st.markdown("### Metrics & Stats")
    st.caption("Upload a labeled CSV to compute a classification report, confusion matrix, and ROC-AUC.")

    up_eval = st.file_uploader("Upload labeled CSV", type=["csv"], key="eval_csv")
    if up_eval:
        df = pd.read_csv(up_eval)
        use_source = None if mission_choice == "auto" else mission_choice
        tgt_col = autodetect_target_col(df, use_source) or st.text_input("Label column name (optional)")
        if tgt_col and tgt_col in df.columns:
            y_raw = df[tgt_col].map(norm_label_generic)
            mask = y_raw.notna()
            dfe = df[mask].copy(); y_true = y_raw[mask].values

            labels, P, used = predict_router(dfe, use_source, koi, k2b, tess, strategy=strategy)
            enc = class_index_map()
            y_true_idx = enc.transform(y_true)
            y_pred_idx = enc.transform(labels)

            st.markdown("---")
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(COMMON_ORDER)))
            cm_df = pd.DataFrame(cm, index=COMMON_ORDER, columns=COMMON_ORDER)
            st.dataframe(cm_df)

            st.markdown("#### Classification Report")
            show_clean_classification_report(
                                            y_true_idx,
                                            y_pred_idx,
                                            class_names=COMMON_ORDER,
                                            proba=P,  # you already have class probabilities from predict_router
                                            key_prefix="eval"
                                        )

            # try:
            #     roc = roc_auc_score(y_true_idx, P, multi_class="ovr")
            #     st.markdown(f"**ROC-AUC (macro OVR):** {roc:.4f}")
            # except Exception:
            #     st.info("ROC-AUC needs valid labels for all classes and probability outputs.")

            st.markdown("#### Probability Distributions")
            st.bar_chart(pd.Series(labels).value_counts())
        else:
            st.info("No label column detected. Type it above if you know it.")

# ===== 4) Explainability =====
with tabs[3]:
    st.markdown("### Explainability")
    st.caption("Feature importances from the current expert (switch mission in the sidebar).")

    use_source = None if mission_choice == "auto" else mission_choice
    bundle = {"kepler": koi, "k2": k2b, "tess": tess}.get(use_source, tess)
    model, le, feat_names, _ = bundle
    try:
        fi = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_}) \
                .sort_values("importance", ascending=False)
        st.markdown("#### Top features")
        st.bar_chart(fi.set_index("feature").head(25))
        st.dataframe(fi.head(50))
    except Exception as e:
        st.warning(f"Could not compute importances: {e}")

# ===== 5) Ingest & Retrain =====
with tabs[4]:
    st.markdown("### Ingest & Retrain")
    st.caption("Retrain a chosen mission expert on uploaded data with hyperparameter tweaking.")

    st.info(
    "• A label column is **recommended** for supervised retraining.\n\n"
    "• No labels ? Enable **Pseudo-labeling** in the sidebar; only high-confidence predictions will be used.\n"
    "• Recommended minimum: **≥ 20** labeled or confident rows.\n"
    "• Features are aligned to the chosen expert’s schema; missing values are median-imputed.\n\n"
    "**Auto-detected label columns:**\n"
    "• **Kepler:** `koi_disposition`, `Disposition Using Kepler Data`\n"
    "• **K2:** `disposition`, `Archive Disposition`, `archive_disposition`\n"
    "• **TESS:** `tfopwg_disp`, `TFOPWG Disposition`, `TFOPWG_DISP`, `disposition`"
    )

    use_source = None if mission_choice == "auto" else mission_choice
    retrain_mission = st.radio("Choose expert to retrain", ["kepler", "k2", "tess"], index=2, horizontal=True)
    expert = normalize_bundle({"kepler": koi, "k2": k2b, "tess": tess}[retrain_mission])
    model_old, le_old, feat_names, imp_old = expert

    up_train = st.file_uploader("Upload CSV for retraining", type=["csv"], key="train_csv")
    if up_train:
        file_tag = f"_{getattr(up_train, 'name', 'nofile')}"
        df = pd.read_csv(up_train)
        tgt_col = autodetect_target_col(df, retrain_mission) or st.text_input("Label column name (optional)", key="lbl2")

        if tgt_col and tgt_col in df.columns:
            y_raw = df[tgt_col].map(norm_label_generic)
            mask = y_raw.notna()
            df = df[mask].copy(); y_raw = y_raw[mask]

            if retrain_mission == "kepler":
                df_proc = preprocess_koi(df.copy())
            elif retrain_mission == "k2":
                df_proc = preprocess_k2(df.copy())
            else:  # "tess"
                df_proc = preprocess_tess(df.copy())
            # Align to the expert schema
            X = df_proc.reindex(columns=feat_names, fill_value=np.nan)
            
            imp = IdentityImputer()  # keep schema; LightGBM handles NaNs
            le = LabelEncoder().fit(COMMON_ORDER)
            y = le.transform(y_raw.values)

            # Button-gated training
            if st.button("🔁 Train / Retrain now", key=f"btn_train_manual_{retrain_mission}{file_tag}"):
                start_time = time.time()
                with st.status("🔁 Training EVRA LightGBM…", expanded=True) as status:
                    status.write("📦 Preparing data (split, align, schema)…")
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    status.write("🚀 Fitting model (this can take a moment)…")
                    new_model = lgb.LGBMClassifier(
                        objective="multiclass",
                        num_class=len(COMMON_ORDER),
                        learning_rate=float(lr),
                        n_estimators=int(nest),
                        num_leaves=int(leaves),
                        min_child_samples=int(minchild),
                        subsample=float(subs),
                        colsample_bytree=float(cols),
                        reg_alpha=float(rega),
                        reg_lambda=float(regl),
                        class_weight=("balanced" if balance else None),
                        random_state=42
                    )
                    new_model.fit(Xtr, ytr)

                    status.update(label="🧪 Validating (20% split)…", state="running")
                    yhat = new_model.predict(Xte)

                    elapsed = time.time() - start_time
                    status.update(label=f"✅ Training & validation complete in {elapsed:.1f}s", state="complete")

                # Show metrics after the status block completes
                show_clean_classification_report(
                    y_true_idx=yte,
                    y_pred_idx=yhat,
                    class_names=COMMON_ORDER,
                    proba=None,
                    title="Validation (20% split)",
                    key_prefix=f"val_{retrain_mission}"
                )

                # Save the trained bundle to session so user can replace/download
                st.session_state["last_retrained_manual"] = {
                    "mission": retrain_mission,
                    "bundle": (new_model, le, feat_names, imp),
                }

        else:
            st.warning("Label column not found. Add it to the CSV or type its name.")

        # If a manual retrain happened in this tab, offer export + replace buttons
        if "last_retrained_manual" in st.session_state and st.session_state["last_retrained_manual"]["mission"] == retrain_mission:
            _model, _le, _f, _imp = st.session_state["last_retrained_manual"]["bundle"]
            buf = BytesIO()
            safe_bundle = serialize_bundle_for_disk((_model, _le, _f, _imp))
            joblib.dump(safe_bundle, buf)
            st.download_button(
                "⬇️ Download retrained model (.pkl)",
                help="You can download the retrained ML model, place it in *models* folder and change path of tess model in config.json file and use it !!!",
                data=buf.getvalue(),
                file_name=f"{retrain_mission}_retrained.pkl"
            )

            # Button (not checkbox) to replace
            # if st.button(
            #             "⬆️ Replace in-session expert with this retrained model",
            #             key=f"btn_replace_manual_{retrain_mission}{file_tag}",
            #             help=(
            #                 "Train on the uploaded data using the current hyperparameters in the controls-sidebar.\n\n"
            #                 "The retrained model is kept in memory only.\n\n"
            #                 "Replace in-session expert: swaps the model only for this session; it does not overwrite files or update config.json.\n\n"
            #                 "Refreshing the page, opening a new tab, clearing cache, or restarting the app restores the original model.\n\n"
            #                 "To make it permanent, download the .pkl and update config.json to point to it."
            #                 )
            #             ):
            #     st.session_state["expert_overrides"][retrain_mission] = (_model, _le, _f, _imp)
            #     st.success("Expert replaced for this session.")

# ===== 6) Model Management =====

with tabs[5]:
    st.markdown("### Model Management")
    st.caption("Check bundle paths, verify availability, and export session models.")
    c1, c2, c3 = st.columns(3)
    with c1: st.code(Path(koi_path).resolve())
    with c2: st.code(Path(k2_path).resolve())
    with c3: st.code(Path(tess_path).resolve())

    # Export current in-memory experts as ZIP of three PKLs, keeping names from config.json
    if st.button("Prepare all current ML models to download in zip format"):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # KOI
            b1 = BytesIO()
            joblib.dump(serialize_bundle_for_disk(koi), b1)
            zf.writestr(Path(koi_path).name, b1.getvalue())

            # K2
            b2 = BytesIO()
            joblib.dump(serialize_bundle_for_disk(k2b), b2)
            zf.writestr(Path(k2_path).name, b2.getvalue())

            # TESS
            b3 = BytesIO()
            joblib.dump(serialize_bundle_for_disk(tess), b3)
            zf.writestr(Path(tess_path).name, b3.getvalue())

        # Important: move the pointer to the beginning before reading
        buf.seek(0)

        st.download_button(
            label="⬇️ Save experts.zip",
            data=buf.getvalue(),
            file_name="EVRA_Experts.zip",
            mime="application/zip",
        )

# ===== 7) About =====
with tabs[6]:
    st.markdown("### About EVRA")
    st.write("""
            **EVRA (Exoplanet Validation & Research with AI)** is a mission-aware, Mixture-of-Experts classifier for NASA exoplanet datasets:
            **Kepler (KOI)**, **K2**, and **TESS**.  
            It predicts **FALSE POSITIVE**, **CANDIDATE**, or **CONFIRMED**, and provides a full workflow:
            **upload → predict → evaluate → (optionally) retrain → export models** — with mission-specific preprocessing and feature engineering preserved.
            """)
    st.markdown("---")
    st.markdown("#### 🧠 How it works")
    st.markdown("""
                - **Mixture-of-Experts (MoE):** EVRA maintains **three experts** (Kepler/K2/TESS).  
                A **router** decides which expert to use:
                - **Overlap (default):** Picks the expert whose **expected features** best overlap with your input (after mission-specific preprocessing).
                - **Soft-vote:** Runs **all** experts and averages their predicted probabilities.
                - You can also **force a mission** (Kepler/K2/TESS) from the sidebar.
                - **Prediction pipeline per expert**:
                1. Apply that mission’s **training-time preprocessing** (drops & feature engineering).
                2. **Align columns** to the expert’s expected `feature_names`.
                3. **Impute** (pass-through for LightGBM; models handle NaNs).
                4. Get **class probabilities** and a **final label** (with optional Candidate thresholding).
                - **Classes:** `FALSE POSITIVE`, `CANDIDATE`, `CONFIRMED` (internal order: FP → Candidate → Confirmed).
                - **Candidate threshold (optional):** If the top class is `CANDIDATE` but its **max probability** is below your **threshold**, it is **downgraded** to `FALSE POSITIVE`.
                """)
    st.markdown("---")
    st.markdown("#### 🧬 Mission-specific preprocessing & feature engineering")
    st.info("EVRA re-applies the **same training-time preprocessing** at inference and during retraining, so feature layouts match each expert’s model exactly.")

    st.markdown("**Kepler (KOI) — Drops & Engineered Features**")
    st.markdown("""
            - **Drop columns:** `kepid`, `kepoi_name`, `kepler_name`, `koi_tce_delivname`, `koi_pdisposition`, `koi_vet_date`, `koi_vet_stat`, `koi_vet_url`
            - **Engineered features (as in training):**
            - `log_prad = log1p(koi_prad)`
            - `log_depth = log1p(koi_depth)`
            - `duration_period_ratio = koi_duration / koi_period`
            - `prad_srad_ratio = koi_prad / koi_srad`
            """)

    st.markdown("**K2 — Drops only (no extra FE beyond your recipe)**")
    st.markdown("""
                - **Drop columns:** `pl_name`, `hostname`, `disp_refname`, `discoverymethod`, `disc_facility`, `soltype`,
                `pl_refname`, `st_refname`, `sy_refname`, `rastr`, `decstr`, `rowupdate`, `pl_pubdate`, `releasedate`,
                `default_flag`, `pl_bmassprov`
                """)

    st.markdown("**TESS — Drops & Engineered Features**")
    st.markdown("""
                - **Drop columns:** `toi`, `tid`, `rastr`, `decstr`, `rowupdate`, `toi_created`
                - **Engineered features (as in training):**
                - **Logs:** `log_pl_orbper`, `log_pl_trandep`, `log_pl_rade`, `log_pl_insol`, `log_st_dist`, `log_st_teff`, `log_pl_eqt`
                - **Ratios/proxies:**  
                    `dur_per_ratio = pl_trandurh / pl_orbper`  
                    `rade_over_st_rad = pl_rade / st_rad`  
                    `pm_total = sqrt(st_pmra^2 + st_pmdec^2)`  
                    `insol_eqt_ratio = log1p(pl_insol) / log1p(pl_eqt)`  
                    `star_density_proxy ≈ st_mass / (st_rad^3)`  
                    `depth_flux_scaled = pl_trandep * 10^(-0.4 * st_tmag)` (rough SNR proxy)
                - **Uncertainty features (relative errors):**  
                    `relerr_pl_orbper`, `relerr_pl_trandurh`, `relerr_pl_trandep`, `relerr_pl_rade`, `relerr_st_teff`, `relerr_st_dist`, `relerr_st_tmag`
                """)
    st.markdown("---")
    st.markdown("#### 📊 Metrics & evaluation")
    st.markdown("""
                - **Metrics & Stats** tab lets you upload **labeled** CSVs and shows:
                - **Confusion matrix** (as a table).
                - **Classification report** (pretty formatted) with **Accuracy (%), Macro-F1, Weighted-F1**.
                - **ROC-AUC (macro OVR)** when probabilities & labels allow.
                - **Label handling:** EVRA **normalizes** label text to `FALSE POSITIVE`, `CANDIDATE`, `CONFIRMED`.
                - **Auto-detect label column** per mission:
                - **Kepler:** `koi_disposition`, `Disposition Using Kepler Data`
                - **K2:** `disposition`, `Archive Disposition`, `archive_disposition`
                - **TESS:** `tfopwg_disp`, `TFOPWG Disposition`, `TFOPWG_DISP`, `disposition`
                """)
    st.markdown("---")
    st.markdown("#### 🔁 Ingest & Retrain")
    st.markdown("""
                - **Retrain per mission** with your CSV (ideally labeled). EVRA:
                1. Applies that mission’s **preprocessing & FE**.
                2. Aligns to the expert’s **feature schema**.
                3. Splits data (**80/20 stratified**), trains **LightGBM** with your sidebar hyperparameters.
                4. Shows validation metrics on the 20% split.
                - **Pseudo-labeling (optional):** If your CSV has **no labels**, EVRA can use its own **high-confidence predictions** as **temporary labels** for retraining.
                - Only samples with max probability ≥ your **pseudo-label threshold** are used.
                - Pseudo-labels **exist only in this session**; the original CSV is not modified.
                - Use a **high threshold** (e.g., 0.90–0.95) to avoid reinforcing mistakes.
                - Always validate on a truly labeled set.
                - **Export the retrained model** (`.pkl`) directly from the retrain tab.
                """)
    st.markdown("---")
    st.markdown("#### 📦 Model export & reuse")
    st.markdown("""
                - **Export all experts as a single ZIP** (Model Management tab):
                - EVRA packages the **current in-memory experts** (Kepler/K2/TESS) into a ZIP with **their original filenames** from your `config.json`.
                - **Replace a model permanently**:
                1. Download the `.pkl` from the **Retrain** tab (or the ZIP from **Model Management**).
                2. Move it into your models folder.
                3. Update `config.json` paths.
                4. Restart EVRA (or reload) to use it.
                """)
    st.markdown("---")  
    st.markdown("#### 🧭 Data expectations & tips")
    st.markdown("""
                - Provide columns consistent with your **mission** (Kepler/K2/TESS).  
                EVRA will **apply mission-specific preprocessing** and align to each expert’s expected feature set.
                - You can **force a mission**, use **overlap** routing (default), or **soft-vote** all experts.
                - If using **Candidate thresholding**, EVRA will **downgrade** low-confidence `CANDIDATE` to `FALSE POSITIVE`.
                - For best results:
                - Ensure numeric fields are truly numeric (no stray strings).
                - When retraining, start with balanced or class-weighted settings if labels are skewed.
                """)
    st.markdown("---")
    st.markdown("#### 🔍 Explainability")
    st.markdown("""
                - The **Explainability** tab shows **LightGBM feature importances** for the **currently selected expert**.
                - Importances reflect the expert’s **feature schema** (including engineered columns).
                """)
    st.markdown("---")
    st.markdown("#### ⚙️ Defaults & behavior")
    st.markdown(f"""
                - **Default routing strategy:** **Overlap**  
                - **Classes (internal order):** `{", ".join(list(COMMON_ORDER))}`
                - **Imputation:** EVRA uses a **pass-through imputer** (LightGBM handles `NaN`), preserving exact training schemas.
                - **Session overrides:** Retrained models can replace experts **in-session** (when enabled), or be **downloaded** for permanent use.
                """)
    st.markdown("---")
    st.markdown("#### 📜 Notes")
    st.markdown("""
                - EVRA keeps preprocessing strictly aligned with each expert’s **training recipe** (drops + engineered features).
                - Soft-vote averages probabilities **after** each expert runs on its **own** preprocessed view of your data.
                - Confusion Matrix is displayed as a plain table to keep evaluation lightweight and readable.
                """)

