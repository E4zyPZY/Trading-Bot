import numpy as np
import pandas as pd


def compute_performance_stats(equity_curve: pd.Series) -> dict:
    """
    Basic stats: total return, CAGR, max drawdown, Sharpe (daily).
    Assumes equity_curve is in currency units (e.g. SGD, USD).
    """
    equity_curve = equity_curve.astype(float)
    ret = equity_curve.pct_change().dropna()

    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]

    if start <= 0:
        total_return = np.nan
        cagr = np.nan
    else:
        total_return = end / start - 1
        years = len(equity_curve) / 252  # daily data approx
        if years > 0 and end > 0:
            cagr = (end / start) ** (1 / years) - 1
        else:
            cagr = np.nan

    if len(ret) > 1 and ret.std() > 0:
        sharpe = np.sqrt(252) * ret.mean() / ret.std()
    else:
        sharpe = np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()

    return {
        "total_return": float(total_return) if total_return == total_return else None,
        "cagr": float(cagr) if cagr == cagr else None,
        "sharpe": float(sharpe) if sharpe == sharpe else None,
        "max_drawdown": float(max_dd),
    }


def compute_risk_metrics(returns: pd.Series, confidence_level: float = 0.95) -> dict:
    """
    Compute Value at Risk (VaR) and Expected Shortfall (ES) for portfolio returns.

    Args:
        returns: Daily returns series
        confidence_level: Confidence level for VaR/ES (e.g., 0.95 for 95%)

    Returns:
        Dict with VaR and ES values
    """
    if len(returns) < 30:
        return {"VaR_95": None, "ES_95": None}

    returns = returns.dropna()

    if len(returns) == 0 or returns.std() == 0:
        return {"VaR_95": None, "ES_95": None}

    # Historical VaR (percentile method)
    var_95 = np.percentile(returns, (1 - confidence_level) * 100)

    # Expected Shortfall (average of returns beyond VaR)
    tail_returns = returns[returns <= var_95]
    es_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95

    return {
        "VaR_95": float(var_95),
        "ES_95": float(es_95),
    }
