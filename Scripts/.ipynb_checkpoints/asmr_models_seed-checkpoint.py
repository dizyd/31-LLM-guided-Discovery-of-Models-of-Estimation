"""
Seed models for the ASMR loop, in the v2 contract.

Contract (also stated to the LLM in the prompt):

    NUM_PARAMETERS = <int>
    def model(parameters, cues, ex_cues=None, ex_crit=None) -> pred_crit

* ``parameters`` are **unconstrained reals of order 1**. Any positivity, range or simplex
  constraint is applied *inside* the function (``np.exp``, softplus, softmax). There is no
  ``BOUNDS`` variable any more -- v1's LLM-authored bounds routinely went stale against the
  model they were paired with (see ``srm_model_1p_0_iteration_4.npz``, where a model
  returning ``np.exp(...)`` was fitted under ``BOUNDS = [(-4, 4)] * 11``).
* Predictions are on the **raw** criterion scale (days / years / grams). To keep the
  optimiser well conditioned, models set their own scale from ``ex_crit`` and ``ex_cues``
  statistics rather than expecting the optimiser to travel to ~1000.
* ``sigma`` is not part of the model. The harness appends one hidden noise parameter and
  the granularity reporting layer in ``asmr_likelihood``.

Two bugs from ``models.py`` / ``models_mammals.py`` are fixed here:

* ``np.power(cues - ex_cues, p)`` is only correct for even ``p``; ``np.abs(...) ** p`` is
  used instead.
* the GCM's attention weights are parameterised by ``n_dim - 1`` free logits through a
  softmax rather than ``n_dim`` weights renormalised to sum to ``n_dim``, which left one
  weight unidentified. (``helper_functions_MLE_fitting.num_parameters`` over-counts the GCM
  by one for the same reason.)
"""

PREAMBLE = '''
def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30.0, 30.0)))


def _softmax(x):
    z = np.asarray(x, dtype=float) - np.max(x)
    e = np.exp(z)
    return e / e.sum()
'''


MODEL_CAM = PREAMBLE + '''
NUM_PARAMETERS = 11

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Unconstrained model parameters, of order 1.

    cues : np.ndarray of shape (n_trials, n_dim)
        Feature matrix of stimuli across trials.

    ex_cues : np.ndarray of shape (n_exemplars, n_dim)
        Feature matrix of the learned exemplars.

    ex_crit : np.ndarray of shape (n_exemplars,)
        Criterion values of the learned exemplars.

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, on the raw criterion scale.
    """
    # Cue Abstraction Model: a weighted linear combination of the cues.
    n_dim   = cues.shape[1]
    loc     = np.mean(ex_crit)
    scale   = np.std(ex_crit) + 1e-9
    cue_sd  = np.std(ex_cues, axis=0) + 1e-9

    intercept = loc + parameters[0] * scale
    weights   = parameters[1:n_dim + 1] * scale / cue_sd

    pred_crit = intercept + cues @ weights

    return pred_crit
'''


MODEL_GCM = PREAMBLE + '''
NUM_PARAMETERS = 10

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Unconstrained model parameters, of order 1.

    cues : np.ndarray of shape (n_trials, n_dim)
        Feature matrix of stimuli across trials.

    ex_cues : np.ndarray of shape (n_exemplars, n_dim)
        Feature matrix of the learned exemplars.

    ex_crit : np.ndarray of shape (n_exemplars,)
        Criterion values of the learned exemplars.

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, on the raw criterion scale.
    """
    # Generalized Context Model: a similarity-weighted average of the exemplar criteria.
    n_dim = cues.shape[1]
    p     = 2.0

    # Reference distance, so that parameters[0] == 0 is a sensible sensitivity.
    ex_gap = np.abs(ex_cues[:, None, :] - ex_cues[None, :, :]) ** p
    ex_gap = (ex_gap.sum(axis=-1)) ** (1.0 / p)
    d_ref  = np.median(ex_gap[ex_gap > 0]) + 1e-9

    # Sensitivity: positive by construction.
    c = np.exp(np.clip(parameters[0], -20.0, 20.0)) / d_ref

    # Attention weights: non-negative and summing to n_dim, with n_dim - 1 free logits
    # (the n_dim-th is fixed at 0, which is what makes the set identified).
    logits = np.concatenate([np.zeros(1), parameters[1:n_dim]])
    w      = _softmax(logits) * n_dim

    diff      = np.abs(cues[:, None, :] - ex_cues[None, :, :]) ** p
    distances = ((diff * w[None, None, :]).sum(axis=-1)) ** (1.0 / p)
    sim       = np.exp(-c * distances)

    sim_sum   = np.maximum(sim.sum(axis=1), 1e-12)
    pred_crit = (sim * ex_crit[None, :]).sum(axis=1) / sim_sum

    return pred_crit
'''


MODEL_RULEXJ = PREAMBLE + '''
NUM_PARAMETERS = 22

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Unconstrained model parameters, of order 1.

    cues : np.ndarray of shape (n_trials, n_dim)
        Feature matrix of stimuli across trials.

    ex_cues : np.ndarray of shape (n_exemplars, n_dim)
        Feature matrix of the learned exemplars.

    ex_crit : np.ndarray of shape (n_exemplars,)
        Criterion values of the learned exemplars.

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, on the raw criterion scale.
    """
    # RulEx-J: a rule module and an exemplar module run in parallel and their interim
    # judgements are blended by alpha (alpha = 1 is pure rule, alpha = 0 pure exemplar).
    n_dim = cues.shape[1]
    p     = 2.0

    alpha = 1.0 / (1.0 + np.exp(-parameters[0]))

    # --- rule module (CAM) ---
    loc       = np.mean(ex_crit)
    scale     = np.std(ex_crit) + 1e-9
    cue_sd    = np.std(ex_cues, axis=0) + 1e-9
    intercept = loc + parameters[1] * scale
    weights   = parameters[2:2 + n_dim] * scale / cue_sd
    judge_r   = intercept + cues @ weights

    # --- exemplar module (GCM) ---
    ex_gap = np.abs(ex_cues[:, None, :] - ex_cues[None, :, :]) ** p
    ex_gap = (ex_gap.sum(axis=-1)) ** (1.0 / p)
    d_ref  = np.median(ex_gap[ex_gap > 0]) + 1e-9

    c      = np.exp(np.clip(parameters[2 + n_dim], -20.0, 20.0)) / d_ref
    logits = np.concatenate([np.zeros(1), parameters[3 + n_dim:2 + 2 * n_dim]])
    w      = _softmax(logits) * n_dim

    diff      = np.abs(cues[:, None, :] - ex_cues[None, :, :]) ** p
    distances = ((diff * w[None, None, :]).sum(axis=-1)) ** (1.0 / p)
    sim       = np.exp(-c * distances)
    sim_sum   = np.maximum(sim.sum(axis=1), 1e-12)
    judge_e   = (sim * ex_crit[None, :]).sum(axis=1) / sim_sum

    pred_crit = alpha * judge_r + (1.0 - alpha) * judge_e

    return pred_crit
'''


MODEL_MAPP = PREAMBLE + '''
NUM_PARAMETERS = 3

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Unconstrained model parameters, of order 1.

    cues : np.ndarray of shape (n_trials, n_dim)
        Feature matrix of stimuli across trials.

    ex_cues : np.ndarray of shape (n_exemplars, n_dim)
        Feature matrix of the learned exemplars.

    ex_crit : np.ndarray of shape (n_exemplars,)
        Criterion values of the learned exemplars.

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, on the raw criterion scale.
    """
    # Mapping model: reduce the object to a single average cue value, then read off the
    # typical criterion value of nearby exemplars. This is the smooth (kernel) version of
    # the manuscript's equally spaced bins: the bin width becomes a bandwidth, which keeps
    # the model differentiable for gradient-based fitting.
    # Cues are first recoded so each is positively related to the criterion.
    crit_c = ex_crit - np.mean(ex_crit)
    signs  = np.sign((ex_cues - np.mean(ex_cues, axis=0)).T @ crit_c)
    signs  = np.where(signs == 0, 1.0, signs)

    ex_avg = (ex_cues * signs[None, :]).mean(axis=1)
    pr_avg = (cues    * signs[None, :]).mean(axis=1)

    spread = np.std(ex_avg) + 1e-9
    bw     = spread * np.exp(np.clip(parameters[0], -20.0, 20.0))          # bandwidth ("bin width")

    d   = (pr_avg[:, None] - ex_avg[None, :]) / bw
    k   = np.exp(-0.5 * d ** 2)
    k   = k / np.maximum(k.sum(axis=1, keepdims=True), 1e-12)

    typical   = k @ ex_crit
    pred_crit = np.mean(ex_crit) + parameters[1] * np.std(ex_crit) \\
                + (1.0 + parameters[2]) * (typical - np.mean(ex_crit))

    return pred_crit
'''


MODEL_CONSTANT = PREAMBLE + '''
NUM_PARAMETERS = 1

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Unconstrained model parameters, of order 1.

    cues : np.ndarray of shape (n_trials, n_dim)
        Feature matrix of stimuli across trials.

    ex_cues : np.ndarray of shape (n_exemplars, n_dim)
        Feature matrix of the learned exemplars.

    ex_crit : np.ndarray of shape (n_exemplars,)
        Criterion values of the learned exemplars.

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, on the raw criterion scale.
    """
    # Reference model: the same prediction on every trial, so it uses no information about
    # the item at all. Under the shared reporting layer this is the honest "response style
    # only" null -- whatever a real model gains over this is what knowing the item buys.
    loc   = np.mean(ex_crit) + parameters[0] * np.std(ex_crit)
    return np.full(cues.shape[0], loc)
'''


SEED_MODELS = {
    "CAM":     MODEL_CAM,
    "GCM":     MODEL_GCM,
    "RulEx-J": MODEL_RULEXJ,
    "MAPP":    MODEL_MAPP,
}

# Reference points, not starting points for discovery.
REFERENCE_MODELS = {
    "Constant": MODEL_CONSTANT,
}


if __name__ == "__main__":
    import numpy as np

    from asmr_codegen import validate_model_code
    from asmr_data import load_aligned

    ds = load_aligned("Mammals")
    for name, src in {**SEED_MODELS, **REFERENCE_MODELS}.items():
        rep = validate_model_code(src, ds.cues, ds.ex_cues, ds.ex_crit)
        print(f"{name:>8}: {'OK ' if rep.ok else 'FAIL'} "
              f"k={rep.num_parameters}  {rep.error or ''}")
        if rep.ok:
            pred = rep.model_fn(np.zeros(rep.num_parameters),
                                ds.cues, ds.ex_cues, ds.ex_crit)
            print(f"{'':>10}pred at theta=0: mean={pred.mean():8.1f} "
                  f"sd={pred.std():7.1f} range=[{pred.min():.0f}, {pred.max():.0f}]"
                  f"   (data mean={np.nanmean(ds.estimates):.0f})")
