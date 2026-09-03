"""
Response model for the ASMR pipeline: a granularity-mixture reporting layer.

Centaur's NLL is a probability **mass** over a single number token. A (trunc)normal
density is not comparable to that, because people report round numbers: in the Mammals
test phase 80.9% of estimates are multiples of 10 and 47.5% are multiples of 100, and
each participant uses only ~37 distinct values across 68 trials. A continuous density
spreads its mass over the whole range and is therefore penalised by a roughly constant
~3.6 nats/trial that has nothing to do with the cognitive process.

This module wraps a deterministic prediction ``mu`` in a reporting stage that emits a
proper PMF over the response grid:

    P(y) = sum_g pi_g * 1[y is a multiple of g] * P_g(y)
    P_g(y) = [Phi((y + g/2 - mu)/sigma) - Phi((y - g/2 - mu)/sigma)] / Z_g

Grid-g bins tile ``[-g/2, N_g*g + g/2]`` exactly, so ``Z_g`` is closed form and each
``P_g`` sums to one over its own support; with ``sum_g pi_g = 1`` the mixture is a proper
PMF over the union of the grids.

``pi_g`` is fitted from each participant's **training-phase** responses, so it costs no
free parameters in the test-phase fit.

Contract for model functions (the LLM writes these):

    def model(parameters, cues, ex_cues=None, ex_crit=None) -> pred_crit

``parameters`` are **unconstrained reals of order 1**; any positivity or range constraint
is applied inside the function. Predictions are on the raw criterion scale.
"""

import numpy as np
from scipy.special import log_ndtr

GRIDS      = (1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0)
SIGMA_MIN  = 1e-6
_LOG_EPS   = -700.0


class ModelEvaluationError(RuntimeError):
    """A candidate model function failed to produce usable predictions.

    Raised -- never swallowed into a sentinel NLL. The v1 pipeline returned ``1e12`` per
    trial here, which is how a crashed model became ``nll.sum() = 6.8e13`` and got
    reported as "the AIC exploded".
    """


def softplus(x):
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return np.log1p(np.exp(x))


# ---------------------------------------------------------------- granularity weights

def coarsest_grid(y, grids=GRIDS, tol=1e-8):
    """Index of the coarsest grid in ``grids`` that ``y`` lies on."""
    y = np.asarray(y, dtype=float)
    out = np.zeros(y.shape, dtype=int)
    for k, g in enumerate(grids):
        on = np.abs(y / g - np.round(y / g)) < tol
        out = np.where(on, k, out)
    return out


def fit_granularity_weights(train_responses, grids=GRIDS, alpha=1.0):
    """Fit ``pi_g`` from a participant's training-phase responses.

    Each response is assigned to the coarsest grid it lies on and the counts are
    Laplace-smoothed. Fitted on training data only, so these are not free parameters of
    the test-phase fit. Falls back to a smoothing-only prior when a participant has no
    usable training responses.
    """
    y = np.asarray(train_responses, dtype=float)
    y = y[np.isfinite(y)]
    counts = np.full(len(grids), float(alpha))
    if y.size:
        idx = coarsest_grid(y, grids)
        counts += np.bincount(idx, minlength=len(grids))
    return counts / counts.sum()


# --------------------------------------------------------------------------- the PMF

def _log_ndtr_diff(a, b):
    """log(Phi(b) - Phi(a)) for b >= a, numerically stable in both tails."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo, hi = np.minimum(a, b), np.maximum(a, b)

    # Lower-tail form where the mass sits left of zero, upper-tail form otherwise;
    # this keeps the subtracted ratio away from 1.
    use_upper = (lo + hi) > 0.0
    x_big  = np.where(use_upper, -lo, hi)
    x_smll = np.where(use_upper, -hi, lo)

    log_big  = log_ndtr(x_big)
    log_smll = log_ndtr(x_smll)
    ratio    = np.clip(log_smll - log_big, None, -1e-12)
    return log_big + np.log1p(-np.exp(ratio))


def grid_masks(y, grids=GRIDS, tol=1e-8):
    """``(n_grids, n)`` membership masks: which responses lie on which grid.

    These depend only on the data, so they are computed once per participant and reused
    across every objective evaluation instead of being recomputed inside the optimiser
    loop.
    """
    y = np.asarray(y, dtype=float)
    m = np.stack([np.abs(y / g - np.round(y / g)) < tol for g in grids])
    # A response on no grid at all (a non-integer estimate) would leave the mixture with
    # no support. Treat it as lying on the finest grid.
    m[0] |= ~m.any(axis=0)
    return m


def logpmf_reported(y, mu, sigma, pi, ub, grids=GRIDS, dist="normal", tol=1e-8,
                    masks=None):
    """Log PMF of the reported response ``y`` under the granularity mixture.

    Parameters
    ----------
    y     : (n,)  observed responses
    mu    : (n,)  model predictions on the raw criterion scale
    sigma : float positive noise scale (on the raw scale, or on the log scale when
            ``dist="lognormal"``)
    pi    : (n_grids,) granularity weights, summing to one
    ub    : float upper bound of the response support
    dist  : "normal" (constant absolute noise) or "lognormal" (Weber-style
            proportional noise, appropriate when responses span orders of magnitude)
    """
    y     = np.asarray(y, dtype=float)
    mu    = np.asarray(mu, dtype=float)
    sigma = max(float(sigma), SIGMA_MIN)
    pi    = np.asarray(pi, dtype=float)

    log_pi = np.where(pi > 0, np.log(np.maximum(pi, 1e-300)), _LOG_EPS)
    masks  = grid_masks(y, grids, tol) if masks is None else masks

    if dist == "lognormal":
        loc = np.log(np.maximum(mu, 1e-6))
        edge = lambda v: np.log(np.maximum(v, 1e-6))          # noqa: E731
        lo_support = 1e-6
    elif dist == "normal":
        loc = mu
        edge = lambda v: v                                    # noqa: E731
        lo_support = -np.inf
    else:
        raise ValueError(f"unknown dist {dist!r}")

    # Everything is evaluated for all grids at once: with only a handful of grids and
    # trials the arrays are tiny, and two vectorised `_log_ndtr_diff` calls are far
    # cheaper than 2 per grid inside a Python loop (this is the optimiser's hot path).
    g   = np.asarray(grids, dtype=float)[:, None]        # (G, 1)
    loc = loc[None, :]                                   # (1, n)

    half = g / 2.0
    log_bin = _log_ndtr_diff(
        (edge(np.maximum(y[None, :] - half, lo_support)) - loc) / sigma,
        (edge(y[None, :] + half) - loc) / sigma,
    )
    log_z = _log_ndtr_diff(
        (edge(np.maximum(-half, lo_support)) - loc) / sigma,
        (edge(np.floor(ub / g) * g + half) - loc) / sigma,
    )

    comps = np.where(masks, log_pi[:, None] + log_bin - log_z, -np.inf)

    top = comps.max(axis=0)
    return top + np.log(np.exp(comps - top[None, :]).sum(axis=0))


# ------------------------------------------------------------------------ NLL wrapper

def predict(model_fn, theta, cues, ex_cues, ex_crit):
    """Call a candidate model and insist that it behaved.

    Raises ``ModelEvaluationError`` rather than returning a sentinel.
    """
    try:
        pred = model_fn(np.asarray(theta, dtype=float), cues, ex_cues, ex_crit)
    except Exception as exc:                       # noqa: BLE001 - untrusted LLM code
        raise ModelEvaluationError(f"{type(exc).__name__}: {exc}") from exc

    pred = np.asarray(pred, dtype=float)
    if pred.shape != (cues.shape[0],):
        raise ModelEvaluationError(
            f"expected predictions of shape {(cues.shape[0],)}, got {pred.shape}"
        )
    if not np.all(np.isfinite(pred)):
        raise ModelEvaluationError("predictions contain NaN or inf")
    return pred


def nll_reported(theta, model_fn, y, cues, ex_cues, ex_crit, pi, ub,
                 sigma_scale=1.0, dist="normal", agg="sum", mask=None, masks=None):
    """Negative log-likelihood of ``y`` under ``model_fn`` plus the reporting layer.

    ``theta[:-1]`` go to the model; ``theta[-1]`` is the hidden noise parameter, which the
    LLM never sees (mirroring how ``sigma`` was hidden in v1). It is unconstrained and
    mapped through ``sigma_scale * softplus(.)`` so that ``theta[-1] ~ 0`` gives a sensible
    sigma and BFGS stays well conditioned.

    ``agg="none"`` returns per-trial NLL with masked trials set to ``nan``.
    """
    theta = np.asarray(theta, dtype=float)
    sigma = float(sigma_scale) * float(softplus(theta[-1])) + SIGMA_MIN
    pred  = predict(model_fn, theta[:-1], cues, ex_cues, ex_crit)

    m = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if masks is None:
        masks = grid_masks(y)
    out = np.full(len(y), np.nan)
    out[m] = -logpmf_reported(y[m], pred[m], sigma, pi, ub, dist=dist,
                              masks=masks[:, m])

    if not np.all(np.isfinite(out[m])):
        raise ModelEvaluationError("non-finite log-likelihood")
    return float(out[m].sum()) if agg == "sum" else out


# ------------------------------------------------------------------------- self-tests

def _check_pmf_sums_to_one(ub=10000.0, grids=GRIDS, dist="normal"):
    """Sum the PMF over the full support; it must come to one."""
    pi = np.array([0.15, 0.10, 0.25, 0.10, 0.25, 0.10, 0.05])
    support = np.unique(np.concatenate(
        [np.arange(0, np.floor(ub / g) + 1) * g for g in grids]
    ))
    worst = 0.0
    for mu in (50.0, 700.0, 3000.0):
        for sigma in ((0.3, 0.8) if dist == "lognormal" else (80.0, 600.0, 2500.0)):
            lp = logpmf_reported(support, np.full(support.size, mu), sigma, pi, ub,
                                 grids=grids, dist=dist)
            worst = max(worst, abs(np.exp(lp).sum() - 1.0))
    return worst


if __name__ == "__main__":
    for dist in ("normal", "lognormal"):
        err = _check_pmf_sums_to_one(dist=dist)
        print(f"{dist:>10}: max |sum P(y) - 1| = {err:.3e}  "
              f"{'OK' if err < 1e-6 else 'FAIL'}")

    from asmr_data import load_aligned

    ds = load_aligned("Mammals")
    pis = [fit_granularity_weights(t) for t in ds.train_est]
    print("\nmean granularity weights over participants "
          f"(grids {[int(g) for g in GRIDS]}):")
    print(np.round(np.mean(pis, axis=0), 3))
