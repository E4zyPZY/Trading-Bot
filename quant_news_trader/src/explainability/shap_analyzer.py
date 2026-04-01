"""
SHAP-based Model Explainability Analysis
=========================================

Explains PCA+RF model predictions through SHAP decomposition:
- Feature importance ranking (which factors matter most?)
- Non-linear factor interactions (e.g., sentiment × volatility)
- Sentiment analysis: negative news impact vs positive news impact on volatility
- Decision plots: how features drive individual predictions
- Summary plots: aggregate feature effects across all predictions

Example:
    from src.explainability.shap_analyzer import SHAPAnalyzer
    
    analyzer = SHAPAnalyzer(model, feature_names)
    analyzer.fit(X_train, y_train)
    analyzer.plot_summary(X_test, "feature_importance.png")
    
    # Analyze sentiment impact specifically
    sentiment_impact = analyzer.analyze_sentiment_impact(X_test, y_test)
    print(f"Negative news volatility coefficient: {sentiment_impact['neg_vol_impact']}")
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.warning("SHAP not installed. Install with: pip install shap")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP-based explainability for PCA+RF trading model.
    
    Attributes:
        model: Trained sklearn model (RandomForest or similar)
        feature_names: List of feature column names
        explainer: SHAP TreeExplainer instance
        shap_values: Computed SHAP values for reference dataset
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained sklearn model with predict/predict_proba
            feature_names: Column names matching model's training features
        """
        if not HAS_SHAP:
            raise ImportError("Install SHAP: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        
        logger.info(f"Initialized SHAP analyzer for {len(feature_names)} features")
    
    def fit(self, X_train: pd.DataFrame, y_train: Optional[np.ndarray] = None, 
            max_samples: int = 100):
        """
        Fit SHAP explainer on training data (background dataset).
        
        Args:
            X_train: Feature matrix for background (smaller is faster)
            y_train: Optional labels (unused but kept for API compatibility)
            max_samples: Subsample background data for speed (100-300 typical)
        
        Returns:
            self for chaining
        """
        # Use background data for SHAP (subset for speed)
        if len(X_train) > max_samples:
            background_idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_background = X_train.iloc[background_idx]
            logger.info(f"Using {max_samples} background samples from {len(X_train)}")
        else:
            X_background = X_train
        
        # Create TreeExplainer (fast, works with tree-based models)
        self.explainer = shap.TreeExplainer(self.model)
        self.base_value = self.explainer.expected_value
        
        logger.info("SHAP explainer fitted successfully")
        return self
    
    def explain(self, X: pd.DataFrame, class_index: int = 1) -> np.ndarray:
        """
        Compute SHAP values for new dataset.
        
        Args:
            X: Feature matrix
            class_index: For binary classification, which class? (0=DOWN, 1=UP)
        
        Returns:
            SHAP values array (n_samples, n_features)
        """
        if self.explainer is None:
            raise ValueError("Call fit() first")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # TreeExplainer returns (n_samples, n_features, n_classes) for multi-class
        # or (n_samples, n_features) for binary with list of classes
        shap_vals = self.explainer.shap_values(X_array)
        
        # Handle different SHAP output formats
        if isinstance(shap_vals, list):
            # Binary classification returns list of arrays
            return shap_vals[class_index]  # Return SHAP for class 1 (UP)
        elif shap_vals.ndim == 3:
            # Multi-dimensional output (samples, features, classes)
            return shap_vals[:, :, class_index]
        else:
            # Standard (samples, features) format
            return shap_vals
    
    def plot_summary(self, X: pd.DataFrame, output_path: str = "shap_summary.png",
                     plot_type: str = "bar", max_display: int = 15):
        """
        Create SHAP summary plot (bar or violin).
        
        Args:
            X: Feature matrix for explanation
            output_path: Where to save plot
            plot_type: "bar" (mean |SHAP|) or "beeswarm" (distribution)
            max_display: Top N features to show
        """
        shap_values = self.explain(X)
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == "bar":
            # Bar plot: mean absolute SHAP value
            feature_importance = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(feature_importance)[-max_display:]
            
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), 
                      [self.feature_names[i] for i in sorted_idx])
            plt.xlabel("Mean |SHAP value|")
            plt.title("SHAP Feature Importance (Mean Impact on Predictions)")
            
        elif plot_type == "beeswarm":
            # Beeswarm: distribution of SHAP values
            feature_importance = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(feature_importance)[-max_display:]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(sorted_idx))
            
            for i, feat_idx in enumerate(sorted_idx):
                ax.scatter([shap_values[:, feat_idx]] * len(shap_values[:, feat_idx]),
                          [i] * len(shap_values[:, feat_idx]),
                          alpha=0.5, s=20)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([self.feature_names[i] for i in sorted_idx])
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title("SHAP Value Distribution by Feature")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {output_path}")
        plt.close()
    
    def plot_dependence(self, X: pd.DataFrame, feature_name: str,
                       output_path: str = "shap_dependence.png"):
        """
        Create SHAP dependence plot: how one feature's value affects its SHAP impact.
        
        Args:
            X: Feature matrix
            feature_name: Which feature to analyze
            output_path: Where to save plot
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found. Available: {self.feature_names}")
        
        feature_idx = self.feature_names.index(feature_name)
        shap_values = self.explain(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X.iloc[:, feature_idx], shap_values[:, feature_idx], alpha=0.6)
        plt.xlabel(f"{feature_name} (feature value)")
        plt.ylabel(f"SHAP value for {feature_name}")
        plt.title(f"SHAP Dependence Plot: {feature_name}")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved dependence plot to {output_path}")
        plt.close()
    
    def analyze_sentiment_impact(self, X: pd.DataFrame, y: np.ndarray,
                                 sentiment_features: Optional[List[str]] = None) \
            -> Dict[str, float]:
        """
        Compare negative vs positive news volatility impact.
        
        Identifies features like 'daily_sentiment_ema_5', 'compound_sentiment', etc.
        Separates baseline volatility effect from sentiment-modulated effect.
        
        Args:
            X: Feature matrix
            y: Labels (1=UP, 0=DOWN)
            sentiment_features: Column names containing sentiment (auto-detect if None)
        
        Returns:
            Dict with:
            - 'neg_vol_impact': Volatility impact when sentiment < 0
            - 'pos_vol_impact': Volatility impact when sentiment > 0
            - 'sentiment_elasticity': (pos - neg) / neg
            - 'feature_interactions': Dict of (sentiment_feat, volatility_feat) interactions
        """
        shap_values = self.explain(X)
        
        # Auto-detect sentiment features
        if sentiment_features is None:
            sentiment_features = [f for f in self.feature_names 
                                 if 'sentiment' in f.lower()]
        
        if not sentiment_features:
            logger.warning("No sentiment features found. Specify with sentiment_features=")
            return {}
        
        results = {}
        
        # For each sentiment feature, analyze volatility interaction
        for sent_feat in sentiment_features:
            if sent_feat not in self.feature_names:
                continue
            
            sent_idx = self.feature_names.index(sent_feat)
            
            # Split by sentiment polarity
            X_negative = X[X.iloc[:, sent_idx] < 0]
            X_positive = X[X.iloc[:, sent_idx] > 0]
            
            if len(X_negative) == 0 or len(X_positive) == 0:
                continue
            
            # Get SHAP values for volatility features
            vol_features = [f for f in self.feature_names 
                           if 'vol' in f.lower() or 'atr' in f.lower()]
            
            for vol_feat in vol_features:
                if vol_feat not in self.feature_names:
                    continue
                
                vol_idx = self.feature_names.index(vol_feat)
                shap_vol = self.explain(X)[:, vol_idx]
                
                # Average SHAP value for vol feature in negative vs positive sentiment
                shap_neg_vol = shap_vol[X.iloc[:, sent_idx] < 0].mean()
                shap_pos_vol = shap_vol[X.iloc[:, sent_idx] > 0].mean()
                
                if shap_neg_vol != 0:
                    elasticity = (shap_pos_vol - shap_neg_vol) / abs(shap_neg_vol)
                else:
                    elasticity = 0
                
                key = f"{sent_feat}_{vol_feat}"
                results[key] = {
                    'negative_impact': shap_neg_vol,
                    'positive_impact': shap_pos_vol,
                    'elasticity': elasticity
                }
        
        logger.info(f"Sentiment impact analysis: {len(results)} interactions found")
        return results
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return feature importance as DataFrame (sorted by mean |SHAP|).
        
        Args:
            X: Feature matrix
        
        Returns:
            DataFrame with columns: feature, importance, rank
        """
        shap_values = self.explain(X)
        
        # Handle both 1D and 2D SHAP arrays
        if shap_values.ndim == 1:
            importance = np.abs(shap_values)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        # Ensure importance is 1D
        importance = np.asarray(importance).flatten()
        
        # Ensure we have the right number of features
        if len(importance) != len(self.feature_names):
            logger.warning(f"Importance shape {len(importance)} != feature count {len(self.feature_names)}")
            importance = importance[:len(self.feature_names)]
        
        df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        df['rank'] = df.index + 1
        
        return df
    
    def explain_prediction(self, x: pd.Series, output_path: Optional[str] = None) \
            -> Dict[str, float]:
        """
        Explain a single prediction using force plot (text format).
        
        Args:
            x: Single feature row (pd.Series)
            output_path: Optional path to save force plot
        
        Returns:
            Dict with feature contributions
        """
        x_array = x.values.reshape(1, -1)
        shap_vals = self.explain(pd.DataFrame([x.values], columns=self.feature_names))[0]
        
        # Ensure shap_vals is 1D
        if shap_vals.ndim > 1:
            shap_vals = shap_vals.flatten()
        
        # Sort by absolute contribution
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_vals,
            'feature_value': x.values,
        }).sort_values('shap_value', ascending=False)
        
        logger.info(f"Prediction explanation for {x.name}:\n{contributions}")
        
        if output_path:
            contributions.to_csv(output_path, index=False)
            logger.info(f"Saved explanation to {output_path}")
        
        return contributions.set_index('feature').to_dict('index')


def compare_model_behaviors(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           model_old, model_new,
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Compare SHAP explainability between old and new models.
    
    Shows which factors changed importance after model upgrade.
    
    Args:
        X_train: Training data
        X_test: Test data
        model_old: Previous model (e.g., NN-based)
        model_new: New model (e.g., PCA+RF)
        feature_names: Feature column names
    
    Returns:
        DataFrame comparing feature importances
    """
    analyzer_new = SHAPAnalyzer(model_new, feature_names)
    analyzer_new.fit(X_train, max_samples=100)
    
    importance_new = analyzer_new.get_feature_importance(X_test)
    importance_new.columns = ['feature', 'importance_new', 'rank_new']
    
    try:
        analyzer_old = SHAPAnalyzer(model_old, feature_names)
        analyzer_old.fit(X_train, max_samples=100)
        importance_old = analyzer_old.get_feature_importance(X_test)
        importance_old.columns = ['feature', 'importance_old', 'rank_old']
        
        comparison = importance_new.merge(importance_old, on='feature', how='outer')
        comparison['importance_delta'] = comparison['importance_new'] - comparison['importance_old']
        comparison['rank_change'] = comparison['rank_old'] - comparison['rank_new']
        
        return comparison.sort_values('importance_delta', ascending=False)
    
    except Exception as e:
        logger.error(f"Could not compare models: {e}")
        return importance_new


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create synthetic data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:100], y[:100])
    
    # Explain predictions
    analyzer = SHAPAnalyzer(model, X.columns.tolist())
    analyzer.fit(X[:50], max_samples=30)
    analyzer.plot_summary(X[100:], output_path="/tmp/shap_summary.png")
    
    feature_importance = analyzer.get_feature_importance(X[100:])
    print(feature_importance)
    
    logger.info("SHAP analysis complete!")
