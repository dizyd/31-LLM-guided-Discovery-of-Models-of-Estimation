import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.optimize import minimize
import numpy.random as rng
from scipy.special import log_ndtr


from configurations import DOMAIN_CONFIG, OPTIM_CONFIG

GRIDS      = (1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0)
SIGMA_MIN  = 1e-6
_LOG_EPS   = -700.0

def load_data(domain):
    """
    Load design data and behavioural judgements for a given domain.

    Returns
    -------
    cues      : (n_test_items, n_dim)   – test-item cue matrix
    ex_cues   : (n_exemplars, n_dim)    – exemplar cue matrix
    ex_crit   : (n_exemplars,)          – exemplar criterion values
    test_ids  : (n_test_items,) int     – 1-indexed test item IDs
    data      : (n_test_items, n_subs)  – float32 participant judgements
    """
    cfg      = DOMAIN_CONFIG[domain]
    n_dim    = cfg["n_dim"]
    cue_cols = [f"V{i}" for i in range(1, n_dim + 1)]

    # ---- Design data ----
    df        = pd.read_csv(cfg["stim"], sep=";", decimal=",")

    exemplars = df.loc[df["training"] == 1]
    ex_cues   = exemplars[cue_cols].to_numpy(dtype=float)
    ex_crit   = exemplars["crit"].to_numpy(dtype=float)
    ex_ids    = exemplars["ID"].to_numpy(dtype=int)

    testing   = df.loc[df["training"] == 0]
    test_ids  = testing["ID"].to_numpy(dtype=int)
    cues      = testing[cue_cols].to_numpy(dtype=float)

    # ---- Behavioural data ----
    data = pd.read_csv(cfg["data"], sep=",").to_numpy()
    # rows = items, columns = [metadata..., sub0, sub1, ...]
    data = np.float32(data[(test_ids - 1), 5:])   # (n_items, n_subs)

    print(f"[{domain}] test items: {cues.shape[0]}, "
          f"exemplars: {ex_cues.shape[0]}, participants: {data.shape[1]}")

    return cues, ex_cues, ex_crit, ex_ids, test_ids, data



def num_parameters(model_name, n_dim):
    """
    Determine the number of parameters for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    n_dim : int
        Number of dimensions in the cue matrix.

    Returns
    -------
    int
        Number of parameters for the specified model.
    """

  
    if model_name == 'CAM':
        return 1 + n_dim + 1           # intercept + n weights + sigma
    elif model_name == 'GCM':
        return 1 + n_dim + 1           # c + n weights + sigma
    else:
        raise ValueError(f"Unknown model: {model_name}")
    


# Define bounds for optimization
def get_bounds(model_name, n_dim, ub):
    """
    Return scipy-style bounds list for the chosen model.

    Parameters
    ----------
    model_name : str    – 'CAM' | 'GCM' 
    n_dim      : int    – number of cue dimensions
    ub         : float  – upper bound for criterion

    Returns
    -------
    list of (low, high) tuples  (None = unconstrained)
    """
    if model_name == 'CAM':
        # [intercept, w_1..w_n, sigma]
        return [(-ub/3, ub/3)] * (n_dim + 1) + [(1e-3, ub)]

    elif model_name == 'GCM':
        # [c, w_1..w_n, sigma]
        return [(1e-4, 100)] + [(0, n_dim)] * n_dim + [(1e-3, ub)]

    else:
        raise ValueError(f"Unknown model: {model_name}")
    


def truncnorm_logpdf(x, mu, sigma, low=0.0, high=100.0):
    """
    Log-PDF of a truncated normal with support [low, high].

    Parameters
    ----------
    x     : np.ndarray  – observed responses
    mu    : np.ndarray  – predicted criterion (location)
    sigma : float       – standard deviation (> 0)
    low, high : float   – truncation bounds

    Returns
    -------
    log_pdf : np.ndarray of shape (n_trials,)
    """
    sigma = max(sigma, 1e-6)
    a = (low  - mu) / sigma
    b = (high - mu) / sigma
    return truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)


def nll_truncnorm(parameters, model_fn, x, cues, ex_cues, ex_crit, ub, agg='sum'):
    """
    Compute negative log-likelihood under a truncated-normal response model.

    Parameters
    ----------
    parameters : np.ndarray   – model parameters passed to model_fn
    model_fn   : callable     – one of model_CAM / model_GCM / model_RULEXJ / model_MAPP
    x          : (n_trials,)  – observed responses
    cues       : (n_trials, n_dim)
    ex_cues    : (n_exemplars, n_dim) or None
    ex_crit    : (n_exemplars,)       or None
    agg        : 'sum' | 'none'       – aggregate or return per-trial NLL

    Returns
    -------
    scalar (agg='sum') or (n_trials,) array (agg='none')
    """
    try:
        pred_crit, sigma = model_fn(parameters, cues, ex_cues, ex_crit)
    except Exception:
        return 1e12 if agg == 'sum' else np.full(len(x), 1e12)

    logpdf = truncnorm_logpdf(x, pred_crit, sigma, high=ub)
    logpdf = np.clip(logpdf, -1e12, None)          # guard -inf

    if agg == 'sum':
        return -logpdf.sum()
    return -logpdf

def nll_truncnorm_woS(parameters, model_fn, x, cues, ex_cues, ex_crit, agg='sum'):
    # The last parameter is ALWAYS the hidden sigma. The LLM never knows about it.
    model_params = parameters[:-1]
    sigma        = abs(parameters[-1])
    
    try:
        # The LLM model now ONLY returns the predictions
        pred_crit = model_fn(model_params, cues, ex_cues, ex_crit)
    except Exception:
        return 1e12 if agg == 'sum' else np.full(len(x), 1e12)

    # Calculate logpdf normally using our hidden sigma
    logpdf = norm.logpdf(x, pred_crit, sigma)
    logpdf = np.clip(logpdf, -1e12, None)          # guard -inf

    if agg == 'sum':
        return -logpdf.sum()
    return -logpdf

# with grid response 
def nll_truncnorm_woS_grid(parameters, model_fn, x, cues, ex_cues, ex_crit, pi, ub, agg='sum',
                           dist='normal'):
    # The last parameter is ALWAYS the hidden sigma. The LLM never knows about it.
    model_params = parameters[:-1]
    sigma        = abs(parameters[-1])
    sentinel     = 1e12 if agg == 'sum' else np.full(len(x), 1e12)

    try:
        # The LLM model now ONLY returns the predictions
        pred_crit = np.asarray(model_fn(model_params, cues, ex_cues, ex_crit), dtype=float)
    except Exception:
        return sentinel
    # Overflowing predictions (e.g. exp of a large linear term) are not a fit, they are a crash.
    if pred_crit.shape != x.shape or not np.all(np.isfinite(pred_crit)):
        return sentinel

    # Calculate logpdf normally using our hidden sigma
    logp = logpmf_reported(x, pred_crit, sigma, pi, ub, dist=dist)
    # A log PMF can never be positive; if it is, the normaliser lost precision (mu far outside
    # the response range). Report that as a crash rather than as a spuriously good fit.
    if np.any(logp > 1e-9):
        return sentinel
    logp = np.clip(logp, -1e12, None)         # guard -inf

    if agg == 'sum':
        return -logp.sum()
    return -logp


def fit_participant(x, model_name, model_fn, cues, ex_cues, ex_crit, n_dim, ub, inits_fn, optim_config=OPTIM_CONFIG):
    """
    Fit a single participant's estimates via MLE with multiple random restarts.

    Parameters
    ----------
    x          : (n_trials,)  – observed responses
    model_name : str
    model_fn   : callable
    cues       : (n_trials, n_dim)
    ex_cues    : (n_exemplars, n_dim) or None
    ex_crit    : (n_exemplars,)       or None
    n_dim      : int

    Returns
    -------
    best_params : np.ndarray
    best_nll    : float
    trial_nll   : (n_trials,)
    """
    bounds      = get_bounds(model_name, n_dim, ub)
    rng_gen     = np.random.default_rng()

    best_nll    = np.inf
    best_params = None

    for _ in range(optim_config["N_RESTARTS"]):
        x0 = inits_fn(model_name, n_dim)

        try:
            res = minimize(
                nll_truncnorm,
                x0,
                args=(model_fn, x, cues, ex_cues, ex_crit, ub, 'sum'),
                method=optim_config["METHOD"],
                bounds=bounds,
                options={'maxiter': optim_config["MAXITER"], 'ftol': optim_config["FTOL"], 'gtol': optim_config["GTOL"]},
            )
            if res.fun < best_nll:
                best_nll    = res.fun
                best_params = res.x
        except Exception as e:
            print("Optimization failed:", e)
            raise

    # Make returned/stored params match the model's effective weights.
    # (The GCM currently normalizes weights internally; we mirror that here so the
    # returned parameter vector respects sum(w)=n_dim.)
    if best_params is not None and model_name == 'GCM':
        w = np.abs(best_params[1:1 + n_dim])
        w = w / (w.sum() + 1e-12) * n_dim
        best_params = best_params.copy()
        best_params[1:1 + n_dim] = w

    trial_nll = nll_truncnorm(best_params, model_fn, x, cues, ex_cues, ex_crit, ub, 'none')
    return best_params, best_nll, trial_nll




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
    ratio    = log_smll - log_big
    # ratio >= 0 means the two tails are numerically identical (the interval carries no
    # representable mass): the answer is log(0), not log_big + log(1e-12) as an old clip
    # produced, which let a model with mu ~ 1e30 score log-probabilities > 0. A nan ratio
    # (inf - inf, from a non-finite mu) is treated the same way.
    with np.errstate(invalid='ignore', divide='ignore'):
        out = log_big + np.log1p(-np.exp(np.minimum(ratio, 0.0)))
    return np.where(np.isfinite(ratio) & (ratio < 0.0), out, -np.inf)


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

    # A component whose normaliser is -inf has no representable mass on the support at all;
    # drop it instead of forming -inf - (-inf).
    comps = np.where(masks & np.isfinite(log_z), log_pi[:, None] + log_bin - log_z, -np.inf)

    top = comps.max(axis=0)
    with np.errstate(invalid='ignore'):
        out = top + np.log(np.exp(comps - top[None, :]).sum(axis=0))
    return np.where(np.isfinite(top), out, -np.inf)