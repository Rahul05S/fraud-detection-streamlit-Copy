import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------------------
# App configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "liquid_fraud_improved_experiment.pt"
SCALER_PATH = APP_DIR / "scaler_improved.joblib"
CONFIG_PATH = APP_DIR / "preprocessing_config_improved.json"

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_FEATURES = [f"type_{name}" for name in TRANSACTION_TYPES]


# -------------------------------------------------------------------
# Original model architecture from the training notebook
# -------------------------------------------------------------------
class LiquidRecurrentCell(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1, unfold_steps=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.unfold_steps = unfold_steps
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size)
        self.raw_tau = nn.Parameter(torch.zeros(hidden_size))
        self.A = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        signal = torch.tanh(
            self.input_layer(x) + self.recurrent_layer(h)
        )
        rate = F.softplus(signal)
        tau = F.softplus(self.raw_tau) + 0.01

        for _ in range(self.unfold_steps):
            numerator = h + self.dt * signal * self.A
            denominator = 1.0 + self.dt * (1.0 / tau + rate)
            h = numerator / denominator

        return h


class LiquidFraudClassifier(nn.Module):
    def __init__(self, input_size=7, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.liquid = LiquidRecurrentCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)
        return self.classifier(h).squeeze(-1)


# -------------------------------------------------------------------
# Load model assets once and reuse them across Streamlit reruns
# -------------------------------------------------------------------
@st.cache_resource
def load_assets():
    missing = [
        path.name
        for path in (MODEL_PATH, SCALER_PATH, CONFIG_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required asset file(s): "
            + ", ".join(missing)
            + ". Put all three files in the same folder as app.py."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    input_size = int(checkpoint["input_size"])
    hidden_size = int(checkpoint["hidden_size"])
    sequence_length = int(checkpoint["sequence_length"])
    threshold = float(checkpoint["threshold"])

    model = LiquidFraudClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    feature_columns = list(checkpoint["feature_columns"])

    if input_size != len(feature_columns):
        raise ValueError(
            "The saved model's input size does not match its feature list."
        )
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != input_size:
        raise ValueError(
            "The saved scaler's feature count does not match the model."
        )

    return model, scaler, config, feature_columns, sequence_length, threshold


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink: #e8eef8;
        --muted: #9baec6;
        --panel: #101d31;
        --panel2: #14243b;
        --line: rgba(163, 184, 208, .17);
        --teal: #35d0bd;
        --blue: #72a7ff;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(36, 120, 150, .20), transparent 35%),
          radial-gradient(ellipse at 88% 8%, rgba(67, 93, 173, .16), transparent 30%),
          #091321;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(9, 19, 33, .72); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1420px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #f2f6fc !important; letter-spacing: -.025em; }
    p, label, .stMarkdown { color: #d7e1ef; }
    .hero {
        border: 1px solid var(--line);
        background: linear-gradient(120deg, rgba(19, 43, 65, .96), rgba(17, 29, 49, .96));
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 22px;
        box-shadow: 0 18px 50px rgba(0,0,0,.16);
    }
    .eyebrow {
        color: var(--teal);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title { font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.05; font-weight: 800; margin: 0; color: #f5f8ff; }
    .hero-copy { color: #b7c7da; font-size: 1.02rem; margin-top: 12px; max-width: 800px; line-height: 1.6; }
    .status-pill {
        display: inline-block; border: 1px solid rgba(53,208,189,.35);
        background: rgba(53,208,189,.10); color: #7ce8d8;
        border-radius: 999px; padding: 7px 12px; font-size: .78rem; font-weight: 700;
        margin-top: 16px;
    }
    .metric-card {
        border: 1px solid var(--line);
        background: linear-gradient(145deg, rgba(20,36,59,.98), rgba(14,27,45,.98));
        border-radius: 18px; padding: 18px 19px; min-height: 126px;
    }
    .metric-label { color: #9fb1c9; font-size: .82rem; font-weight: 650; }
    .metric-value { color: #f4f8ff; font-size: 1.9rem; font-weight: 800; margin-top: 9px; letter-spacing: -.04em; }
    .metric-note { color: #8299b5; font-size: .76rem; margin-top: 3px; }
    .section-heading { font-size: 1.22rem; font-weight: 750; color: #f2f6fc; margin: 8px 0 5px 0; }
    .section-copy { color: #9fb1c9; margin-bottom: 16px; font-size: .92rem; }
    .upload-panel {
        border: 1px dashed rgba(114,167,255,.45); background: rgba(18,35,57,.70);
        border-radius: 20px; padding: 18px 20px;
    }
    .small-muted { color: #9fb1c9; font-size: .86rem; }
    .risk-high { color: #ff8c9a; font-weight: 800; }
    .risk-normal { color: #76e1c4; font-weight: 750; }
    div[data-testid="stFileUploader"] {
        background: rgba(13, 26, 43, .55); border-radius: 14px; padding: 10px;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #20b9ac, #3c9eea);
        border: 0; border-radius: 12px; color: #071523; font-weight: 800;
        min-height: 3rem; box-shadow: 0 8px 24px rgba(32,185,172,.18);
        transition: transform .16s ease, filter .16s ease;
    }
    div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
    div.stDownloadButton > button {
        border-radius: 11px; border: 1px solid rgba(53,208,189,.45);
        background: rgba(53,208,189,.10); color: #a7f3e8; font-weight: 750;
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 14px; }
    hr { border-color: var(--line); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_features(frame, feature_columns):
    """Build the exact seven features expected by the saved scaler/model."""
    normalized_names = [str(col).strip().lower() for col in frame.columns]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError(
            "Your CSV has duplicate column names after ignoring case/spaces. "
            "Please rename the duplicate columns and upload again."
        )

    lookup = {normalized: original for normalized, original in zip(normalized_names, frame.columns)}
    required = ["step", "amount", "type"]
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(
            "This CSV is missing required column(s): "
            + ", ".join(missing)
            + ". The demo expects a PaySim-compatible CSV containing step, amount, and type."
        )

    cleaned = frame.copy()
    cleaned["_original_order"] = np.arange(len(cleaned))
    cleaned["_step_numeric"] = pd.to_numeric(cleaned[lookup["step"]], errors="coerce")
    cleaned["_amount_numeric"] = pd.to_numeric(cleaned[lookup["amount"]], errors="coerce")
    cleaned["_type_clean"] = cleaned[lookup["type"]].astype("string").str.strip().str.upper()

    if cleaned["_step_numeric"].isna().any():
        raise ValueError("The step column contains blank or non-numeric values.")
    if cleaned["_amount_numeric"].isna().any():
        raise ValueError("The amount column contains blank or non-numeric values.")
    if (cleaned["_amount_numeric"] < 0).any():
        raise ValueError("The amount column contains negative values; this demo expects non-negative transaction amounts.")
    if not np.isfinite(cleaned["_step_numeric"].to_numpy(dtype=float)).all():
        raise ValueError("The step column contains non-finite values.")
    if not np.isfinite(cleaned["_amount_numeric"].to_numpy(dtype=float)).all():
        raise ValueError("The amount column contains non-finite values.")

    observed_types = set(cleaned["_type_clean"].dropna().astype(str).unique())
    if cleaned["_type_clean"].isna().any() or "" in observed_types:
        raise ValueError("The type column contains blank transaction types.")
    unknown_types = sorted(observed_types - set(TRANSACTION_TYPES))
    if unknown_types:
        raise ValueError(
            "Unsupported transaction type(s): "
            + ", ".join(unknown_types)
            + ". This model was trained on: "
            + ", ".join(TRANSACTION_TYPES)
            + "."
        )

    # The training data was sorted chronologically. Keep original order as a
    # stable tie-breaker when multiple transactions share the same step.
    cleaned = cleaned.sort_values(
        ["_step_numeric", "_original_order"], kind="stable"
    ).reset_index(drop=True)

    features = pd.DataFrame(index=cleaned.index)
    features["step"] = cleaned["_step_numeric"].astype(float)

    # The saved scaler column is named "amount", but training used log1p(amount).
    features["amount"] = np.log1p(cleaned["_amount_numeric"].astype(float))

    for transaction_type in TRANSACTION_TYPES:
        features[f"type_{transaction_type}"] = (
            cleaned["_type_clean"] == transaction_type
        ).astype(float)

    # Force exact training feature order, not whatever order a CSV uses.
    features = features.reindex(columns=feature_columns)
    if features.isna().any().any():
        raise ValueError(
            "The generated feature columns do not match the saved model configuration."
        )

    return cleaned, features


def score_transactions(model, scaler, feature_frame, sequence_length, threshold):
    """Predict each row from the preceding sequence_length transactions."""
    scaled = scaler.transform(feature_frame).astype(np.float32)
    row_count = len(scaled)

    probabilities = np.full(row_count, np.nan, dtype=np.float64)
    if row_count <= sequence_length:
        return probabilities

    model.eval()
    batch_size = 512
    with torch.inference_mode():
        # Training used X[i:i+sequence_length] to predict y[i+sequence_length].
        # Therefore the first sequence_length rows are context only.
        for first_target in range(sequence_length, row_count, batch_size):
            last_target = min(first_target + batch_size, row_count)
            windows = np.stack(
                [
                    scaled[target - sequence_length:target]
                    for target in range(first_target, last_target)
                ],
                axis=0,
            )
            tensor = torch.from_numpy(windows)
            logits = model(tensor)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities[first_target:last_target] = batch_probabilities

    return probabilities


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">TRANSACTION INTELLIGENCE · LOCAL DEMO</div>
        <div class="hero-title">Fraud Detection Studio</div>
        <div class="hero-copy">
            Screen transaction streams with your trained liquid recurrent model.
            Upload a compatible CSV, review risk scores, and export the results in a few clicks.
        </div>
        <div class="status-pill">● MODEL READY · CPU INFERENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

dataset_mode = st.radio(
    "Choose dataset format",
    ["PaySim", "BankSim"],
    horizontal=True,
    help="PaySim uses the saved liquid recurrent model. BankSim uses a separate model trained on the uploaded labeled BankSim CSV because its transaction categories and schema differ.",
)

if dataset_mode == "BankSim":
    st.markdown('<div class="section-heading">BankSim dataset analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload a labeled BankSim CSV. This mode trains a separate baseline classifier using a chronological holdout when a step column is available. It does not reuse the PaySim-trained liquid model.</div>',
        unsafe_allow_html=True,
    )
    bank_file = st.file_uploader(
        "Upload BankSim CSV",
        type=["csv"],
        key="banksim_uploader",
        help="Common columns include step, amount, category, age, gender, and fraud. The target label is required for training/evaluation.",
    )
    bank_threshold = st.slider(
        "BankSim flagging threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        key="banksim_threshold",
        help="Lower thresholds usually flag more transactions and may increase false positives.",
    )
    train_bank = st.button("Train and analyze BankSim", type="primary", use_container_width=True, disabled=bank_file is None)

    if train_bank and bank_file is not None:
        try:
            bank_df = pd.read_csv(bank_file)
            if bank_df.empty:
                raise ValueError("The uploaded CSV contains no transaction rows.")
            if len(bank_df) < 50:
                raise ValueError("Please upload at least 50 labeled rows for a meaningful train/test split.")

            bank_lookup = {str(c).strip().lower(): c for c in bank_df.columns}
            label_candidates = ["fraud", "isfraud", "is_fraud", "label", "class", "isfraudulent"]
            label_key = next((name for name in label_candidates if name in bank_lookup), None)
            if label_key is None:
                raise ValueError("Could not find a fraud label column. Expected one of: fraud, isFraud, is_fraud, label, or class.")
            target_raw = bank_df[bank_lookup[label_key]]
            if target_raw.dtype == object or str(target_raw.dtype).startswith("string"):
                mapped = target_raw.astype(str).str.strip().str.lower().map({
                    "0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1,
                    "legitimate": 0, "normal": 0, "fraud": 1,
                })
                y_bank = mapped
            else:
                y_bank = pd.to_numeric(target_raw, errors="coerce")
            valid = y_bank.isin([0, 1])
            bank_df = bank_df.loc[valid].copy().reset_index(drop=True)
            y_bank = y_bank.loc[valid].astype(int).reset_index(drop=True)
            if y_bank.nunique() != 2:
                raise ValueError("The fraud label must contain both classes: 0 (non-fraud) and 1 (fraud).")
            if int(y_bank.value_counts().min()) < 2:
                raise ValueError("There must be at least two examples of each class to create a train/test split.")

            # Use meaningful transaction attributes; exclude identifiers and the target to reduce leakage.
            chosen_keys = [key for key in ["step", "amount", "category", "age", "gender", "type"] if key in bank_lookup and bank_lookup[key] != bank_lookup[label_key]]
            if not chosen_keys:
                raise ValueError("No supported input columns found. Expected at least one of step, amount, category, age, gender, or type.")
            X_bank = bank_df[[bank_lookup[k] for k in chosen_keys]].copy()
            X_bank.columns = chosen_keys
            for col in X_bank.columns:
                if col in {"step", "amount", "age"}:
                    X_bank[col] = pd.to_numeric(X_bank[col], errors="coerce")
                else:
                    X_bank[col] = X_bank[col].astype("string").fillna("Unknown")
            numeric_cols = [c for c in X_bank.columns if c in {"step", "amount", "age"}]
            categorical_cols = [c for c in X_bank.columns if c not in numeric_cols]
            transformers = []
            if numeric_cols:
                transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols))
            if categorical_cols:
                transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols))
            preprocess = ColumnTransformer(transformers=transformers)
            bank_model = Pipeline([
                ("preprocess", preprocess),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
            ])
            step_col = "step" if "step" in X_bank.columns else None
            if step_col:
                order = np.argsort(pd.to_numeric(X_bank[step_col], errors="coerce").fillna(-np.inf).to_numpy(), kind="stable")
                X_bank = X_bank.iloc[order].reset_index(drop=True)
                y_bank = y_bank.iloc[order].reset_index(drop=True)
                split_at = int(len(X_bank) * 0.8)
                X_train, X_test = X_bank.iloc[:split_at], X_bank.iloc[split_at:]
                y_train, y_test = y_bank.iloc[:split_at], y_bank.iloc[split_at:]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_bank, y_bank, test_size=0.2, random_state=42, stratify=y_bank
                    )
                    split_note = "Stratified random holdout (chronological split lacked both classes)."
                else:
                    split_note = "Chronological 80/20 holdout by step."
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_bank, y_bank, test_size=0.2, random_state=42, stratify=y_bank
                )
                split_note = "Stratified random 80/20 holdout (no step column available)."
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                raise ValueError("The split does not contain both fraud classes in train and test. Upload more labeled data covering both classes.")

            with st.spinner("Training the BankSim-specific baseline and evaluating the holdout…"):
                bank_model.fit(X_train, y_train)
                test_prob = bank_model.predict_proba(X_test)[:, 1]
                test_pred = (test_prob >= bank_threshold).astype(int)
                all_prob = bank_model.predict_proba(X_bank)[:, 1]
                all_pred = (all_prob >= bank_threshold).astype(int)

            precision = precision_score(y_test, test_pred, zero_division=0)
            recall = recall_score(y_test, test_pred, zero_division=0)
            f1 = f1_score(y_test, test_pred, zero_division=0)
            ap = average_precision_score(y_test, test_prob)
            auc = roc_auc_score(y_test, test_prob) if y_test.nunique() == 2 else float("nan")
            cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

            st.success(f"BankSim baseline trained. {split_note}")
            st.caption("This is a separate Logistic Regression baseline, not the PaySim liquid recurrent network. Metrics below are calculated on the holdout set, not the training rows.")
            cards = st.columns(5)
            for col, label, value in zip(cards, ["Precision", "Recall", "F1 score", "Average precision", "ROC-AUC"], [precision, recall, f1, ap, auc]):
                with col:
                    metric_card(label, f"{value:.3f}" if np.isfinite(value) else "N/A", "Holdout performance")
            st.markdown("**Holdout confusion matrix**")
            st.dataframe(pd.DataFrame(cm, index=["Actual non-fraud", "Actual fraud"], columns=["Predicted non-fraud", "Predicted fraud"]), use_container_width=True)
            st.caption(f"Rows used: {len(X_bank):,} · Training rows: {len(X_train):,} · Holdout rows: {len(X_test):,} · Fraud label column: `{bank_lookup[label_key]}` · Features: {', '.join(chosen_keys)}")

            bank_results = bank_df.iloc[order if step_col else np.arange(len(bank_df))].copy() if step_col else bank_df.copy()
            bank_results["fraud_probability"] = all_prob
            bank_results["risk_label"] = np.where(all_pred == 1, "Flagged for review", "Lower model score")
            st.markdown("**BankSim transaction results**")
            st.dataframe(bank_results.head(1000), use_container_width=True, hide_index=True)
            st.download_button("Download BankSim scored results", data=bank_results.to_csv(index=False).encode("utf-8-sig"), file_name="banksim_fraud_results.csv", mime="text/csv", use_container_width=True)
            st.info("The displayed scored results include predictions on rows used to fit the model, so use the holdout metrics above for performance reporting. This baseline is for education/research, not live banking decisions.")
        except Exception as exc:
            st.error(f"Could not process BankSim CSV: {exc}")
    st.stop()

try:
    model, scaler, config, feature_columns, sequence_length, threshold = load_assets()
except Exception as exc:
    st.error(f"Could not load the model assets: {exc}")
    st.info(
        "Check that app.py, liquid_fraud_pilot.pt, scaler.joblib, and "
        "preprocessing_config.json are all in the same folder."
    )
    st.stop()

saved_threshold = threshold

with st.expander("Model and input requirements", expanded=False):
    st.markdown(
        f"""
        - **Model:** educational liquid recurrent classifier
        - **Input columns required:** `step`, `amount`, `type`
        - **Transaction types supported:** {", ".join(TRANSACTION_TYPES)}
        - **Sequence length:** {sequence_length} preceding transactions
        - **Saved validation threshold:** `{saved_threshold:.7f}`
        - **Feature preprocessing:** `log1p(amount)`, one-hot transaction type, then the saved scaler
        """
    )
    st.caption(
        "For this prototype, transactions are sorted by step. The first "
        f"{sequence_length} rows provide context and do not receive a prediction."
    )

st.markdown('<div class="section-heading">Decision threshold</div>', unsafe_allow_html=True)
st.caption(
    "The saved threshold was selected on the validation split. This control lets you "
    "explore the precision–recall trade-off for a demo; changing it does not retrain the model."
)
decision_threshold = st.slider(
    "Flag for review when fraud probability is at least",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    format="%.2f",
    help=f"Saved validation threshold: {saved_threshold:.4f}. The demo starts at 0.50 so you can inspect lower-threshold behavior; validate any chosen threshold on separate labeled data.",
)
threshold_delta = decision_threshold - saved_threshold
st.caption(
    f"Current threshold: {decision_threshold:.2f} · Saved validation threshold: {saved_threshold:.4f}"
    + (f" · Current setting is {abs(threshold_delta):.2f} lower than the saved threshold." if threshold_delta < 0 else "")
)

st.markdown('<div class="section-heading">01 / Upload transaction data</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Use a PaySim-compatible CSV for this demo. '
    'Extra columns are retained in the report, but only step, amount, and type are used by the model.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Required columns: step, amount, type. Keep the file in chronological transaction order if possible.",
)

if uploaded_file is not None:
    st.caption(f"Selected file: **{uploaded_file.name}** · {uploaded_file.size / 1024:,.1f} KB")

analyze = st.button(
    "Analyze transactions",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)

if analyze and uploaded_file is not None:
    try:
        with st.spinner("Validating data and scoring transaction sequences…"):
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)

            if raw_df.empty:
                raise ValueError("The uploaded CSV has headers but no transaction rows.")
            if len(raw_df) > 1_000_000:
                raise ValueError(
                    "This local demo is limited to 1,000,000 rows per upload. "
                    "Please upload a smaller CSV sample."
                )

            ordered_df, feature_frame = prepare_features(raw_df, feature_columns)
            probabilities = score_transactions(
                model,
                scaler,
                feature_frame,
                sequence_length,
                decision_threshold,
            )

            results = ordered_df.drop(
                columns=["_original_order", "_step_numeric", "_amount_numeric", "_type_clean"],
                errors="ignore",
            ).copy()
            results["fraud_probability"] = probabilities
            results["risk_label"] = np.where(
                np.isnan(probabilities),
                "Insufficient history",
                np.where(probabilities >= decision_threshold, "Flagged for review", "Lower model score"),
            )
            results["risk_label"] = results["risk_label"].astype(str)

            st.session_state["fraud_results"] = results
            st.session_state["fraud_filename"] = uploaded_file.name
            st.session_state["fraud_threshold"] = decision_threshold
            st.session_state["fraud_sequence_length"] = sequence_length
            st.session_state["fraud_row_count"] = len(results)

    except Exception as exc:
        st.error(f"Could not analyze this file: {exc}")
        st.stop()

results = st.session_state.get("fraud_results")
if results is not None:
    threshold = st.session_state["fraud_threshold"]
    sequence_length = st.session_state["fraud_sequence_length"]
    probabilities = results["fraud_probability"]
    scored_mask = probabilities.notna()
    scored_count = int(scored_mask.sum())
    flagged_mask = scored_mask & (probabilities >= threshold)
    flagged_count = int(flagged_mask.sum())
    lower_count = scored_count - flagged_count
    flagged_percent = (flagged_count / scored_count * 100) if scored_count else 0.0

    st.markdown("---")
    st.markdown('<div class="section-heading">02 / Analysis overview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-copy">Results for <b>{st.session_state.get("fraud_filename", "uploaded CSV")}</b>. '
        'A flag indicates a model score above the current threshold—not confirmed fraud.</div>',
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4, gap="medium")
    with metric_columns[0]:
        metric_card("Transactions uploaded", f"{len(results):,}", "Rows in the input file")
    with metric_columns[1]:
        metric_card("Transactions scored", f"{scored_count:,}", f"First {sequence_length} rows used as context")
    with metric_columns[2]:
        metric_card("Flagged for review", f"{flagged_count:,}", f"{flagged_percent:.2f}% of scored rows")
    with metric_columns[3]:
        metric_card("Lower model score", f"{lower_count:,}", "Below the current threshold")

    if scored_count == 0:
        st.warning(
            f"Upload at least {sequence_length + 1} transactions to score one row. "
            "The first sequence is used as context."
        )
    elif flagged_count:
        st.warning(
            f"{flagged_count:,} transaction(s) exceeded the current threshold. "
            "Review these records carefully; the model does not confirm fraud."
        )
    else:
        st.success("No scored transactions exceeded the current threshold in this upload.")

    st.markdown('<div class="section-heading">03 / Fraud summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A quick view of how the model categorized this upload. '
        'A flag means the transaction deserves review; it does not prove fraud.</div>',
        unsafe_allow_html=True,
    )
    risk_counts = (
        results["risk_label"]
        .value_counts()
        .reindex(["Flagged for review", "Lower model score", "Insufficient history"], fill_value=0)
        .rename_axis("Model assessment")
        .reset_index(name="Transactions")
    )
    summary_chart, summary_details = st.columns([1.35, 1], gap="large")
    with summary_chart:
        st.bar_chart(
            risk_counts.set_index("Model assessment"),
            y="Transactions",
            horizontal=True,
            use_container_width=True,
        )
    with summary_details:
        total_rows = len(results)
        flagged_count = int(flagged_mask.sum())
        scored_count = int(scored_mask.sum())
        summary_metrics = st.columns(2)
        with summary_metrics[0]:
            metric_card("Flagged", f"{flagged_count:,}", f"{flagged_count / scored_count:.2%} of scored rows" if scored_count else "No scored rows")
        with summary_metrics[1]:
            metric_card("Scored", f"{scored_count:,}", f"of {total_rows:,} uploaded rows")
        st.caption(
            "The model score is not a calibrated guarantee of fraud. Use flags to prioritize "
            "human review, and evaluate performance against labeled data when available."
        )

    # Optional post-hoc evaluation: only available when the uploaded CSV contains
    # a usable ground-truth isFraud column (as in labeled PaySim data).
    if "isFraud" in results.columns:
        actual = pd.to_numeric(results["isFraud"], errors="coerce")
        valid_labels = actual.isin([0, 1]) & scored_mask
        eval_count = int(valid_labels.sum())

        st.markdown('<div class="section-heading">04 / Model evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">These metrics compare model flags with the uploaded '
            '`isFraud` labels for scored rows only. They are a sample-level evaluation, not a '
            'guarantee of performance on other data.</div>',
            unsafe_allow_html=True,
        )

        if eval_count:
            y_true = actual.loc[valid_labels].astype(int)
            y_pred = (probabilities.loc[valid_labels] >= threshold).astype(int)

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            accuracy = (tp + tn) / eval_count if eval_count else 0.0

            eval_columns = st.columns(4, gap="medium")
            with eval_columns[0]:
                metric_card("Precision", f"{precision:.2%}", "Of flagged rows, fraction labeled fraud")
            with eval_columns[1]:
                metric_card("Recall", f"{recall:.2%}", "Of labeled fraud rows, fraction flagged")
            with eval_columns[2]:
                metric_card("F1 score", f"{f1:.3f}", "Harmonic mean of precision and recall")
            with eval_columns[3]:
                metric_card("Evaluated rows", f"{eval_count:,}", "Scored rows with valid 0/1 labels")

            confusion_left, confusion_right = st.columns([1, 1.25], gap="large")
            with confusion_left:
                st.markdown("**Confusion matrix**")
                confusion_df = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Actual: not fraud (0)", "Actual: fraud (1)"],
                    columns=["Predicted: lower score (0)", "Predicted: flagged (1)"],
                )
                # Use a plain dataframe to avoid requiring the optional matplotlib package.
                st.dataframe(confusion_df, use_container_width=True)
                st.caption("Counts use the current threshold; the first sequence rows are excluded.")
            with confusion_right:
                st.markdown("**How to read these results**")
                st.markdown(
                    f"- **True positives (TP):** {tp:,} labeled-fraud rows flagged by the model.\n"
                    f"- **False positives (FP):** {fp:,} non-fraud rows flagged for review.\n"
                    f"- **False negatives (FN):** {fn:,} labeled-fraud rows not flagged.\n"
                    f"- **True negatives (TN):** {tn:,} non-fraud rows not flagged."
                )
                st.caption(
                    "Fraud datasets are often highly imbalanced, so accuracy alone can be misleading. "
                    "Review precision, recall, false positives, and the evaluation split together."
                )
        else:
            st.warning(
                "The `isFraud` column has no valid 0/1 labels among scored rows, so evaluation metrics "
                "cannot be calculated."
            )

    st.markdown('<div class="section-heading">05 / Transaction results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Filter and inspect the output below. Rows are shown in chronological step order.</div>',
        unsafe_allow_html=True,
    )

    filter_choice = st.radio(
        "Show transactions",
        ["All transactions", "Flagged for review", "Scored only", "Insufficient history"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if filter_choice == "Flagged for review":
        display_df = results.loc[flagged_mask].copy()
    elif filter_choice == "Scored only":
        display_df = results.loc[scored_mask].copy()
    elif filter_choice == "Insufficient history":
        display_df = results.loc[~scored_mask].copy()
    else:
        display_df = results.copy()

    # Show a useful set of columns first while retaining the full data in downloads.
    preferred_columns = [
        "step", "type", "amount", "fraud_probability", "risk_label",
        "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    visible_columns = [col for col in preferred_columns if col in display_df.columns]
    visible_columns += [
        col for col in display_df.columns
        if col not in visible_columns and col not in {"isFraud", "isFlaggedFraud"}
    ]
    if "isFraud" in display_df.columns:
        visible_columns.append("isFraud")
    if "isFlaggedFraud" in display_df.columns:
        visible_columns.append("isFlaggedFraud")

    styled_df = display_df[visible_columns].copy()
    if "fraud_probability" in styled_df.columns:
        styled_df["fraud_probability"] = styled_df["fraud_probability"].round(6)

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=440,
        column_config={
            "fraud_probability": st.column_config.NumberColumn(
                "Fraud score",
                help="Sigmoid-transformed model output. It is a model score, not a calibrated guarantee of fraud probability.",
                format="%.6f",
            ),
            "risk_label": st.column_config.TextColumn("Model assessment"),
        },
    )

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download full results (CSV)",
        data=csv_bytes,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if "isFraud" in results.columns:
        st.info(
            "The uploaded file includes an `isFraud` column. It is retained for reference only; "
            "the app does not use it to generate predictions."
        )

st.markdown("---")
st.markdown(
    """
    <div class="small-muted">
      <b>Prototype notice:</b> This is an educational research demo using a trained pilot model
      and synthetic-data feature assumptions. A flagged transaction is only a candidate for human review.
      This tool is not validated for live banking, payment authorization, or operational fraud decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
