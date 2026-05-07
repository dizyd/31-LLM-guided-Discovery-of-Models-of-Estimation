import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize
import numpy.random as rng

from configurations import DOMAIN_CONFIG, OPTIM_CONFIG

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
        pred_crit, sigma = model_fn(parameters, cues, ub, ex_cues, ex_crit)
    except Exception:
        return 1e12 if agg == 'sum' else np.full(len(x), 1e12)

    logpdf = truncnorm_logpdf(x, pred_crit, sigma, high=ub)
    logpdf = np.clip(logpdf, -1e12, None)          # guard -inf

    if agg == 'sum':
        return -logpdf.sum()
    return -logpdf



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

