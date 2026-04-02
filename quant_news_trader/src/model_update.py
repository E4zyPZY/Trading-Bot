"""
Model Update System
===================

Automated model improvement with forward-rolling updates:
- Micro updates (3 days): Incremental learning on recent data
- Macro updates (7 days): Full retrain with extended data
- Major updates (30 days): Hyperparameter tuning and architecture optimization
- Epoch updates (365 days): Complete model overhaul and benchmarking

Usage:
    python -m src.model_update --update-type micro
    python -m src.model_update --update-type macro
    python -m src.model_update --update-type major
    python -m src.model_update --update-type epoch
"""

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_for_watchlist
from src.features.sentiment import add_sentiment
from src.features.feature_builder import build_features
from src.models.model import train_pca_direction_model, predict_proba_pca


class ModelUpdater:
    """
    Handles automated model updates with different frequencies and scopes.
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.root = Path(__file__).resolve().parents[1]

        # Create output directories
        self.models_dir = self.root / "models"
        self.reports_dir = self.root / "reports"
        self.logs_dir = self.root / "logs"

        for dir_path in [self.models_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Update configuration
        self.update_cfg = config.get("update", {})
        self.market_cfg = config["market"]
        self.model_cfg = config["model"]

    def _load_current_model(self) -> Optional[Dict]:
        """Load the current production model."""
        model_path = self.models_dir / "current_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load current model: {e}")
        return None

    def _save_model(self, model_dict: Dict, update_type: str):
        """Save updated model with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{update_type}_{timestamp}.pkl"

        model_path = self.models_dir / filename
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)

        # Update current model symlink/reference
        current_path = self.models_dir / "current_model.pkl"
        try:
            current_path.unlink(missing_ok=True)
            current_path.symlink_to(filename)
        except Exception:
            # Fallback: just copy the file
            import shutil
            shutil.copy2(model_path, current_path)

        self.logger.info(f"Saved model: {model_path}")
        return model_path

    def _collect_update_data(self, update_type: str) -> tuple:
        """Collect data for the update based on update type."""
        base_lookback = self.update_cfg.get("base_lookback_days", 180)

        if update_type == "micro":
            # Micro: Recent data + some historical context
            new_data_days = self.update_cfg["micro_update"].get("new_data_days", 30)
            total_lookback = base_lookback + new_data_days
        elif update_type == "macro":
            # Macro: Extended historical window
            extended = self.update_cfg["macro_update"].get("extended_lookback", 365)
            total_lookback = extended
        elif update_type in ["major", "epoch"]:
            # Major/Epoch: Maximum historical data
            total_lookback = base_lookback * 3  # 540 days
        else:
            total_lookback = base_lookback

        # Fetch market data
        self.logger.info(f"Fetching {total_lookback} days of market data...")
        price_df = fetch_market_data(
            ticker=self.market_cfg["ticker"],
            lookback_days=total_lookback,
            interval=self.market_cfg["interval"]
        )

        # Fetch news data
        self.logger.info("Fetching news data...")
        tickers = self.config.get("market", {}).get("tickers") or [self.market_cfg["ticker"]]
        news_df = fetch_news_for_watchlist(
            base_query=self.config["news"].get("query"),
            tickers=tickers,
            language=self.config["news"]["language"],
            lookback_days=min(total_lookback, 90),  # News API limits
            max_articles=self.config["news"]["max_articles"],
            include_macro_news=bool(self.config["news"].get("include_macro_news", True)),
            macro_queries=self.config["news"].get("macro_queries"),
            per_query_article_cap=self.config["news"].get("per_query_article_cap"),
        )

        return price_df, news_df

    def _perform_micro_update(self, current_model: Dict) -> Dict:
        """Micro update: Incremental learning on recent data."""
        self.logger.info("Performing MICRO UPDATE (incremental learning)")

        # Collect recent data
        price_df, news_df = self._collect_update_data("micro")

        # Process data
        news_df = add_sentiment(news_df)
        feat_df = build_features(price_df, news_df)

        if feat_df.empty:
            raise ValueError("No features generated for micro update")

        # Incremental update: combine old and new data
        retain_weight = self.update_cfg["micro_update"].get("retain_old_weight", 0.8)

        # For simplicity, do a partial retrain with weighted data
        # In production, you'd use online learning algorithms
        new_model = train_pca_direction_model(
            feat_df,
            n_components=int(self.model_cfg.get("pca_components", 8)),
            test_size=float(self.model_cfg.get("test_size", 0.2)),
            random_state=int(self.model_cfg.get("random_state", 42)),
            n_jobs=int(self.model_cfg.get("n_jobs", 1)),
        )

        # Log performance improvement
        old_score = current_model.get("report", {}).get("accuracy", 0)
        new_score = new_model.get("report", {}).get("accuracy", 0)

        self.logger.info(".3f"
                        ".3f")

        return new_model

    def _perform_macro_update(self, current_model: Dict) -> Dict:
        """Macro update: Full retrain with extended data."""
        self.logger.info("Performing MACRO UPDATE (full retrain)")

        # Collect extended data
        price_df, news_df = self._collect_update_data("macro")

        # Process data with fresh features
        news_df = add_sentiment(news_df)
        feat_df = build_features(price_df, news_df)

        if feat_df.empty:
            raise ValueError("No features generated for macro update")

        # Full retrain
        new_model = train_pca_direction_model(
            feat_df,
            n_components=int(self.model_cfg.get("pca_components", 8)),
            test_size=float(self.model_cfg.get("test_size", 0.2)),
            random_state=int(self.model_cfg.get("random_state", 42)),
            n_jobs=int(self.model_cfg.get("n_jobs", 1)),
        )

        # Log performance
        old_score = current_model.get("report", {}).get("accuracy", 0)
        new_score = new_model.get("report", {}).get("accuracy", 0)

        self.logger.info(".3f"
                        ".3f")

        return new_model

    def _perform_major_update(self, current_model: Dict) -> Dict:
        """Major update: Hyperparameter tuning."""
        self.logger.info("Performing MAJOR UPDATE (hyperparameter tuning)")

        # Collect data
        price_df, news_df = self._collect_update_data("major")
        news_df = add_sentiment(news_df)
        feat_df = build_features(price_df, news_df)

        if feat_df.empty:
            raise ValueError("No features generated for major update")

        # Hyperparameter tuning
        self.logger.info("Running hyperparameter optimization...")

        # Define parameter space
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [6, 8, 10, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        # Prepare data
        X = feat_df[feat_df.columns.difference(['target_up'])]
        y = feat_df['target_up']

        # Randomized search
        rf = RandomForestClassifier(random_state=42)
        cv_folds = self.update_cfg["major_update"].get("cross_validation_folds", 5)

        search = RandomizedSearchCV(
            rf, param_dist, n_iter=20, cv=cv_folds,
            scoring='accuracy', n_jobs=-1, random_state=42
        )

        search.fit(X, y)

        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(".3f")

        # Create final model with best parameters
        best_rf = RandomForestClassifier(**search.best_params_, random_state=42)

        # Retrain PCA pipeline with tuned RF
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=int(self.model_cfg.get("pca_components", 8)))
        X_pca = pca.fit_transform(X_scaled)

        best_rf.fit(X_pca, y)

        # Package results
        new_model = {
            "model": best_rf,
            "scaler": StandardScaler(),
            "pca": pca,
            "feature_cols": list(X.columns),
            "report": {
                "accuracy": search.best_score_,
                "best_params": search.best_params_,
                "cv_scores": search.cv_results_['mean_test_score'].tolist()
            },
            "update_type": "major",
            "update_time": datetime.now().isoformat()
        }

        return new_model

    def _perform_epoch_update(self, current_model: Dict) -> Dict:
        """Epoch update: Complete model overhaul."""
        self.logger.info("Performing EPOCH UPDATE (complete overhaul)")

        # This would involve:
        # 1. Benchmarking against other model types (XGBoost, Neural Networks, etc.)
        # 2. Major feature engineering updates
        # 3. Architecture changes

        # For now, perform a comprehensive update
        price_df, news_df = self._collect_update_data("epoch")
        news_df = add_sentiment(news_df)
        feat_df = build_features(price_df, news_df)

        # Try different model architectures
        self.logger.info("Benchmarking model architectures...")

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        models = {
            'rf': RandomForestClassifier(n_estimators=300, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }

        X = feat_df[feat_df.columns.difference(['target_up'])]
        y = feat_df['target_up']

        # Scale for SVM and LR
        X_scaled = StandardScaler().fit_transform(X)

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(8, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_pca, y, cv=5, scoring='accuracy')
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores.tolist()
            }
            self.logger.info(f"{name.upper()}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
        best_model = models[best_model_name]

        self.logger.info(f"Selected {best_model_name.upper()} as best architecture")

        # Train final model
        best_model.fit(X_pca, y)

        new_model = {
            "model": best_model,
            "scaler": StandardScaler(),
            "pca": pca,
            "feature_cols": list(X.columns),
            "report": {
                "benchmark_results": results,
                "selected_model": best_model_name,
                "accuracy": results[best_model_name]['mean_accuracy']
            },
            "update_type": "epoch",
            "update_time": datetime.now().isoformat()
        }

        return new_model

    def update_model(self, update_type: str) -> bool:
        """
        Perform model update based on type.

        Args:
            update_type: "micro", "macro", "major", or "epoch"

        Returns:
            True if update successful
        """
        if not self.update_cfg.get("enabled", False):
            self.logger.info("Model updates disabled in config")
            return False

        self.logger.info(f"Starting {update_type.upper()} model update...")

        try:
            # Load current model
            current_model = self._load_current_model()
            if current_model is None and update_type != "epoch":
                self.logger.warning("No current model found, performing full training")
                current_model = {"report": {"accuracy": 0.5}}  # Default baseline

            # Perform update based on type
            if update_type == "micro":
                new_model = self._perform_micro_update(current_model)
            elif update_type == "macro":
                new_model = self._perform_macro_update(current_model)
            elif update_type == "major":
                new_model = self._perform_major_update(current_model)
            elif update_type == "epoch":
                new_model = self._perform_epoch_update(current_model)
            else:
                raise ValueError(f"Unknown update type: {update_type}")

            # Save model
            model_path = self._save_model(new_model, update_type)

            # Generate update report
            self._generate_update_report(new_model, update_type, model_path)

            # Log to update history
            self._log_update_history(update_type, new_model)

            self.logger.info(f"✅ {update_type.upper()} update completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ {update_type.upper()} update failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_update_report(self, model_dict: Dict, update_type: str, model_path: Path):
        """Generate detailed update report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"model_update_{update_type}_{timestamp}.json"

        report = {
            "update_type": update_type,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "performance": model_dict.get("report", {}),
            "configuration": {
                "market": self.market_cfg,
                "model": self.model_cfg,
                "update": self.update_cfg
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Update report saved: {report_path}")

    def _log_update_history(self, update_type: str, model_dict: Dict):
        """Log update to history file."""
        history_path = self.logs_dir / "model_update_history.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "update_type": update_type,
            "performance": model_dict.get("report", {}),
            "model_info": str(type(model_dict.get("model", {})))
        }

        with open(history_path, 'a') as f:
            json.dump(entry, f)
            f.write('\n')

        self.logger.info(f"Update logged to history: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Update System")
    parser.add_argument(
        "--update-type",
        choices=["micro", "macro", "major", "epoch"],
        required=True,
        help="Type of model update to perform"
    )
    parser.add_argument(
        "--profile",
        help="Configuration profile to use"
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root, args.profile)

    # Run update
    updater = ModelUpdater(config, logger)
    success = updater.update_model(args.update_type)

    if success:
        logger.info("🎉 Model update completed successfully!")
        return 0
    else:
        logger.error("💥 Model update failed!")
        return 1


if __name__ == "__main__":
    exit(main())