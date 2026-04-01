from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "rolling_vol_5",
    "rolling_vol_20",
    "daily_sentiment",
    "daily_sentiment_ema_5",
    "daily_sentiment_ema_20",
    "ema_10",
    "ema_50",
    "ema_200",
    "ema_10_50_diff",
    "ema_50_200_diff",
    "ma_20",
    "ma_50",
    "bb_pct",
    "macd_12_26",
    "macd_signal_9",
    "macd_hist",
    "rsi_14",
    "atr_14",
    "obv",
    "swing_low_20",
    "dist_to_swing_low_20",
    "volume_rel_20",
    "stop_run_20",
    "stop_run_hi_vol_20",
    "day_of_week",
    "month",
]


def train_direction_model(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Train a RandomForest to predict next-day up/down.
    Completely local, no API usage.
    """
    X = features_df[FEATURE_COLS]
    y = features_df["target_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False,  # keep time order
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, X_train, X_test, y_train, y_test, report


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def train_pca_direction_model(
    features_df: pd.DataFrame,
    n_components: int = 8,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = 1,
) -> dict:
    """Train an RF model on PCA-transformed quant features."""
    X = features_df[FEATURE_COLS].copy()
    y = features_df["target_up"]

    if X.isna().any().any():
        # simple fill; in production use better imputation
        X = X.fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "feature_cols": FEATURE_COLS,
        "report": report,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def predict_proba_pca(pipeline: dict, row: pd.DataFrame) -> tuple:
    """Predict [prob_down, prob_up] from a single row using PCA model pipeline."""
    if isinstance(row, pd.Series):
        row = row.to_frame().T

    x = row[pipeline["feature_cols"]]
    x = x.fillna(0.0)
    x_scaled = pipeline["scaler"].transform(x)
    x_pca = pipeline["pca"].transform(x_scaled)

    proba = pipeline["model"].predict_proba(x_pca)[0]
    return float(proba[0]), float(proba[1])

