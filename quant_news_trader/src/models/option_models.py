import math
from typing import Dict, Tuple


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bsm_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes–Merton call delta for non-dividend-paying stock.
    S: spot
    K: strike
    T: time to maturity in years
    r: risk-free rate
    sigma: annual volatility
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.5  # shrug: no info, neutral

    d1 = (
        math.log(S / K) + (r + 0.5 * sigma * sigma) * T
    ) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def binomial_call_delta_one_step(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    One-step Cox-Ross-Rubinstein (CRR) binomial call delta.

    We use a single big step over horizon T, which gives an approximate
    delta. For more steps, you'd build a full tree; for our purposes
    this is enough to differentiate from BSM slightly.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.5

    dt = T
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(r * dt)

    # risk-neutral up probability
    p = (disc - d) / (u - d)
    p = max(0.0, min(1.0, p))

    Su = S * u
    Sd = S * d

    Cu = max(Su - K, 0.0)
    Cd = max(Sd - K, 0.0)

    # delta at t=0
    delta = (Cu - Cd) / (Su - Sd) if Su != Sd else 0.5
    # Clamp for safety
    delta = max(0.0, min(1.0, delta))
    return delta


def merton_effective_sigma(
    sigma: float,
    lambda_j: float,
    mu_j: float,
    delta_j: float,
) -> float:
    """
    Very simplified Merton effective volatility.

    In Merton jump–diffusion:
      sigma_eff^2 = sigma^2 + lambda_j * (delta_j^2 + mu_j^2)

    We use that just to get a fatter-tailed effective sigma, then plug
    it into BSM delta. This is a simplification for this project.
    """
    if sigma <= 0:
        return 0.0

    extra_var = lambda_j * (delta_j * delta_j + mu_j * mu_j)
    sigma_eff = math.sqrt(max(1e-12, sigma * sigma + extra_var))
    return sigma_eff


def merton_call_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    delta_j: float,
) -> float:
    """
    Merton jump–diffusion: approximate call delta by using an
    'effective sigma' in a BSM-like delta. This is not a full
    Merton formula, but captures the idea that jumps change
    effective volatility.
    """
    sigma_eff = merton_effective_sigma(sigma, lambda_j, mu_j, delta_j)
    if sigma_eff <= 0:
        return 0.5
    return bsm_call_delta(S, K, T, r, sigma_eff)


def delta_to_score(delta: float) -> float:
    """
    Convert delta in [0,1] to a score in [-1,1].
    >0 => bullish, <0 => bearish.
    """
    return 2.0 * delta - 1.0


def choose_best_action_from_models(
    S: float,
    sigma_annual: float,
    cfg: Dict,
) -> Tuple[float, Dict]:
    """
    For a given spot S and annual volatility sigma_annual, use three models
    (Binomial, BSM, Merton) to propose actions and pick the most confident.

    Returns
    -------
    signal : float in {-1.0, 0.0, +1.0}
        -1 = short, 0 = flat, +1 = long
    details : dict
        Contains deltas, scores and chosen model.
    """
    # Basic config
    maturity_days = float(cfg.get("maturity_days", 30))
    r = float(cfg.get("risk_free_rate", 0.03))
    min_conf = float(cfg.get("min_confidence", 0.1))

    T = maturity_days / 252.0  # years
    K = S  # ATM

    # Merton params
    m_cfg = cfg.get("merton", {})
    lambda_j = float(m_cfg.get("lambda_j", 0.5))
    mu_j = float(m_cfg.get("mu_j", -0.1))
    delta_j = float(m_cfg.get("delta_j", 0.2))

    if S <= 0 or sigma_annual <= 0 or T <= 0:
        return 0.0, {
            "reason": "invalid_inputs",
            "S": S,
            "sigma_annual": sigma_annual,
        }

    # Compute deltas
    delta_bsm = bsm_call_delta(S, K, T, r, sigma_annual)
    delta_binom = binomial_call_delta_one_step(S, K, T, r, sigma_annual)
    delta_merton = merton_call_delta(S, K, T, r, sigma_annual, lambda_j, mu_j, delta_j)

    # Convert to scores in [-1, 1]
    score_bsm = delta_to_score(delta_bsm)
    score_binom = delta_to_score(delta_binom)
    score_merton = delta_to_score(delta_merton)

    # Proposed actions per model
    def sign_or_zero(x: float, eps: float = 1e-6) -> int:
        if x > eps:
            return 1
        elif x < -eps:
            return -1
        else:
            return 0

    action_bsm = sign_or_zero(score_bsm)
    action_binom = sign_or_zero(score_binom)
    action_merton = sign_or_zero(score_merton)

    models = {
        "bsm": {"delta": delta_bsm, "score": score_bsm, "action": action_bsm},
        "binomial": {"delta": delta_binom, "score": score_binom, "action": action_binom},
        "merton": {"delta": delta_merton, "score": score_merton, "action": action_merton},
    }

    # Pick the model with maximum absolute score (confidence)
    best_name = None
    best_score_abs = -1.0
    for name, info in models.items():
        sc = abs(info["score"])
        if sc > best_score_abs:
            best_score_abs = sc
            best_name = name

    if best_name is None or best_score_abs < min_conf:
        # not confident enough => flat
        return 0.0, {
            "chosen_model": None,
            "models": models,
            "reason": "low_confidence",
        }

    chosen_action = float(models[best_name]["action"])
    details = {
        "chosen_model": best_name,
        "models": models,
        "chosen_score": models[best_name]["score"],
    }
    return chosen_action, details
