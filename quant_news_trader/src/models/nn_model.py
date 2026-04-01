# src/models/nn_model.py

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    optim = None
    _HAS_TORCH = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional XGBoost for ensemble
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    XGBClassifier = None
    _HAS_XGB = False


def infer_feature_columns(feat_df: pd.DataFrame) -> List[str]:
    """
    Heuristic: use all numeric columns except obvious non-features / labels.
    """
    exclude = {
        "signal",
        "direction",
        "target",
        "target_up",
        "prob_up",
        "prob_down",
        "equity",
        "chosen_model",
        "chosen_score",
        "chosen_maturity_days",
        # multi-horizon label columns should not be used as features
        "direction_1d",
        "direction_3d",
        "direction_5d",
        "ret_fwd_1d",
        "ret_fwd_3d",
        "ret_fwd_5d",
    }

    cols: List[str] = []
    for c in feat_df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(feat_df[c]):
            cols.append(c)
    return cols


class MLPClassifier(nn.Module if _HAS_TORCH else object):
    def __init__(self, input_dim: int):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required to use MLPClassifier. "
                "Install it with: pip install torch"
            )
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 classes: 0 = down/non-up, 1 = up
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_nn_direction_model(
    feat_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[
    MLPClassifier,
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Train a simple GPU-friendly MLP to predict next-day direction.

    Assumes feat_df has a binary 'direction' column (0/1).
    Returns:
        model         : trained PyTorch MLP
        feature_cols  : list of columns used as features
        X_train, X_test, y_train, y_test
        report        : sklearn classification_report (dict)
        context       : {"feature_cols": ..., "device": "cpu"/"cuda"}
    """
    if "direction" not in feat_df.columns:
        raise ValueError("feat_df must contain a 'direction' column (0/1).")

    # Drop rows with missing label or feature NaNs
    feature_cols = infer_feature_columns(feat_df)
    df = feat_df.dropna(subset=feature_cols + ["direction"]).copy()

    if df.empty:
        raise ValueError("No rows left after dropping NaNs for features + direction.")

    X = df[feature_cols].values.astype(np.float32)
    y = df["direction"].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    num_batches = int(np.ceil(len(X_train_t) / batch_size))

    model.train()
    for epoch in range(num_epochs):
        # simple shuffle
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            xb = X_train_t[start:end].to(device)
            yb = y_train_t[start:end].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Optional logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"[NN] Epoch {epoch+1}/{num_epochs}, loss={avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t.to(device))
        probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
        y_pred = probs_test.argmax(axis=1)

    report = classification_report(y_test, y_pred, output_dict=True)

    context = {
        "feature_cols": feature_cols,
        "device": device.type,
    }

    return model, feature_cols, X_train, X_test, y_train, y_test, report, context


def predict_proba_nn(
    model: MLPClassifier,
    x_row: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict class probabilities for a single feature row.

    x_row: shape (n_features,)
    Returns: probs shape (2,) [p(class0), p(class1)]
    """
    dev = torch.device(device)
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_row.astype(np.float32)).unsqueeze(0).to(dev)
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


# ---------------------------------------------------------------------
# Multi-horizon + Ensemble logic (NN + XGBoost)
# ---------------------------------------------------------------------


def train_ensemble_for_horizon(
    feat_df: pd.DataFrame,
    horizon_days: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    Train:
      - NN classifier (existing MLP)
      - XGBoost classifier (if installed)
    for a given horizon (1,3,5 days).

    Uses label column: f"direction_{horizon_days}d".
    Returns an ensemble dict:
      {
        "horizon_days": ...,
        "nn_model": nn_model,
        "xgb_model": xgb_or_None,
        "feature_cols": feature_cols,
        "device": "cpu"/"cuda",
        "report_nn": ...,
        "report_xgb": ... or None,
      }
    """

    label_col = f"direction_{horizon_days}d"
    if label_col not in feat_df.columns:
        raise ValueError(f"{label_col} not in feat_df. Did you update build_features?")

    # Use horizon-specific label as "direction"
    tmp = feat_df.copy()
    tmp = tmp.dropna(subset=[label_col])
    tmp["direction"] = tmp[label_col]

    (
        nn_model,
        feature_cols,
        X_train,
        X_test,
        y_train,
        y_test,
        report_nn,
        context_nn,
    ) = train_nn_direction_model(
        tmp,
        test_size=test_size,
        random_state=random_state,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    xgb_model = None
    report_xgb = None

    if _HAS_XGB:
        # XGB uses same features & labels
        X = tmp[feature_cols].values.astype("float32")
        y = tmp["direction"].values.astype("int64")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        xgb_model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
        )
        xgb_model.fit(X_tr, y_tr)
        y_pred_xgb = xgb_model.predict(X_te)
        report_xgb = classification_report(y_te, y_pred_xgb, output_dict=True)

    ensemble = {
        "horizon_days": horizon_days,
        "nn_model": nn_model,
        "xgb_model": xgb_model,
        "feature_cols": feature_cols,
        "device": context_nn["device"],
        "report_nn": report_nn,
        "report_xgb": report_xgb,
    }
    return ensemble


def predict_probs_multi_horizon_ensemble(
    ensembles: Dict[int, Dict[str, Any]],
    x_row: np.ndarray,
    long_horizons=(1, 5),
    device_fallback: str = "cpu",
    w_nn: float = 0.5,
    w_xgb: float = 0.5,
):
    """
    ensembles: {horizon_days: ensemble_dict}
    x_row: features, shape (n_features,)

    Returns:
      agg_prob_up, agg_prob_down, detail

    detail = {h: {"p_up": ..., "p_down": ...}, ...}

    Aggregation is conservative: we take min(prob_up) across horizons
    (requires agreement across 1d & 5d views).
    """

    detail: Dict[int, Dict[str, float]] = {}
    per_h_up: List[float] = []
    per_h_down: List[float] = []

    for h in long_horizons:
        ens = ensembles.get(h)
        if ens is None:
            continue

        device = ens.get("device", device_fallback)
        nn_model = ens["nn_model"]
        xgb_model = ens["xgb_model"]

        # NN prob
        p_nn = predict_proba_nn(nn_model, x_row, device=device)
        p0_nn, p1_nn = float(p_nn[0]), float(p_nn[1])

        # XGB prob (if available)
        if xgb_model is not None:
            p_xgb = xgb_model.predict_proba(x_row.reshape(1, -1))[0]
            p0_xgb, p1_xgb = float(p_xgb[0]), float(p_xgb[1])
        else:
            p0_xgb, p1_xgb = p0_nn, p1_nn

        # Weighted average; if w_nn + w_xgb < 1.0, leftover weight reuses NN
        leftover = max(0.0, 1.0 - (w_nn + w_xgb))
        p0 = leftover * p0_nn + w_nn * p0_nn + w_xgb * p0_xgb
        p1 = leftover * p1_nn + w_nn * p1_nn + w_xgb * p1_xgb

        detail[h] = {"p_up": p1, "p_down": p0}
        per_h_up.append(p1)
        per_h_down.append(p0)

    if not per_h_up:
        # fallback: neutral
        return 0.5, 0.5, detail

    # conservative: require all horizons to be strong
    agg_prob_up = min(per_h_up)
    agg_prob_down = min(per_h_down)

    return agg_prob_up, agg_prob_down, detail
