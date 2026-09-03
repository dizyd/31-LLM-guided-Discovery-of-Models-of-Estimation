"""
Maximum-likelihood fitting for the ASMR pipeline.

Differences from v1 (``ASMR.ipynb``):

* multiple random restarts instead of a single start at ``abs(0.01 * randn(k + 1))``, which
  put the GCM at ``c ~ 0.01`` (a flat prediction) every time. ``OPTIM_CONFIG["N_RESTARTS"]``
  was already available in ``configurations.py`` and unused by the ASMR loop;
* unconstrained parameters and plain BFGS, as in the ASMR paper -- no LLM-authored
  ``BOUNDS`` to go stale;
* the raw criterion scale, matching the reporting layer and matching what the prompt says;
* a failed model evaluation is an error, never a ``1e12`` sentinel that silently becomes
  the reported fit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from asmr_likelihood import (GRIDS, ModelEvaluationError, fit_granularity_weights,
                             grid_masks, nll_reported, predict)
from configurations import OPTIM_CONFIG

# Finite penalty used *inside* the objective when a candidate blows up in some region of
# parameter space. It never reaches the reported numbers: the final NLL is recomputed with
# `nll_reported`, which raises rather than returning a sentinel.
OBJECTIVE_PENALTY = 1e8


@dataclass
class Fit:
    """Result of fitting one model form to every participant."""

    theta:      np.ndarray      # (n_subs, k + 1) incl. the hidden noise parameter
    trial_nll:  np.ndarray      # (n_subs, n_items) nats; nan on masked trials
    k:          int             # free parameters per participant, incl. sigma
    ok:         np.ndarray      # (n_subs,) bool
    message:    str = ""

    @property
    def total_nll(self):
        return float(np.nansum(self.trial_nll))

    @property
    def n_valid_trials(self):
        return int(np.isfinite(self.trial_nll).sum())

    @property
    def aic(self):
        """Summed over participants, each with its own parameter set (as in v1)."""
        return 2.0 * self.k * int(self.ok.sum()) + 2.0 * self.total_nll

    @property
    def nats_per_trial(self):
        return self.total_nll / max(self.n_valid_trials, 1)


def granularity_weights(ds, grids=GRIDS):
    """Per-participant granularity weights, fitted on training-phase responses only."""
    return np.array([fit_granularity_weights(t, grids) for t in ds.train_est])


def fit_one(model_fn, k, y, mask, cues, ex_cues, ex_crit, pi, ub,
            sigma_scale, dist="normal", n_restarts=None, rng=None, masks=None):
    """Fit one participant. Returns ``(theta, per_trial_nll)`` or raises.

    ``k`` is the model's ``NUM_PARAMETERS``; one hidden noise parameter is appended.
    """
    rng        = np.random.default_rng() if rng is None else rng
    n_restarts = OPTIM_CONFIG["N_RESTARTS"] if n_restarts is None else n_restarts
    masks      = grid_masks(y) if masks is None else masks

    def objective(theta):
        try:
            return nll_reported(theta, model_fn, y, cues, ex_cues, ex_crit, pi, ub,
                                sigma_scale=sigma_scale, dist=dist, agg="sum",
                                mask=mask, masks=masks)
        except ModelEvaluationError:
            return OBJECTIVE_PENALTY
        except (FloatingPointError, ValueError):
            return OBJECTIVE_PENALTY

    best_theta, best_val = None, np.inf
    for r in range(n_restarts):
        # First restart starts at the origin, which the model contract defines as a
        # sensible default (predictions already on the right scale); the rest explore.
        x0 = np.zeros(k + 1) if r == 0 else rng.normal(0.0, 1.0, size=k + 1)
        try:
            # BFGS, not OPTIM_CONFIG["METHOD"] (L-BFGS-B): parameters are unconstrained
            # by contract, so there are no bounds to enforce.
            res = minimize(objective, x0, method="BFGS",
                           options={"maxiter": OPTIM_CONFIG["MAXITER"],
                                    "gtol": OPTIM_CONFIG["GTOL"]})
        except Exception:                                       # noqa: BLE001
            continue
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val, best_theta = float(res.fun), res.x

    if best_theta is None or best_val >= OBJECTIVE_PENALTY:
        raise ModelEvaluationError("no restart produced a usable fit")

    # Recompute without the penalty escape hatch: this raises if the optimum sits in a
    # region where the model is not actually evaluable.
    trial_nll = nll_reported(best_theta, model_fn, y, cues, ex_cues, ex_crit, pi, ub,
                             sigma_scale=sigma_scale, dist=dist, agg="none",
                             mask=mask, masks=masks)
    return best_theta, trial_nll


_WORKER = {}


def _worker_init(source, cues, ex_cues, ex_crit):
    """Re-create the model function inside a worker process.

    Model functions come from ``exec`` and are therefore not picklable, so workers are
    given the *source* and rebuild it once.
    """
    from asmr_codegen import validate_model_code

    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    rep = validate_model_code(source, cues, ex_cues, ex_crit, n_draws=2)
    if not rep.ok:
        raise RuntimeError(rep.error)
    _WORKER["model_fn"] = rep.model_fn


def _worker_fit(args):
    p, k, y, mask, cues, ex_cues, ex_crit, pi, ub, scale, dist, n_restarts, seed, masks = args
    try:
        th, nll = fit_one(_WORKER["model_fn"], k, y, mask, cues, ex_cues, ex_crit, pi, ub,
                          scale, dist=dist, n_restarts=n_restarts,
                          rng=np.random.default_rng(seed), masks=masks)
    except ModelEvaluationError as exc:
        return p, None, None, str(exc)
    return p, th, nll, None


def fit_all(model_fn, k, ds, pis=None, dist="normal", n_restarts=None, rng=None,
            items=None, seed=0, n_jobs=1, source=None):
    """Fit ``model_fn`` to every participant.

    ``items`` restricts the fit to a subset of test-item rows (used for cross-validation).
    Pass ``n_jobs > 1`` together with ``source`` (the model's code) to fit participants in
    parallel processes; the 48 per-participant fits are independent.
    """
    pis = granularity_weights(ds) if pis is None else pis
    idx = np.arange(ds.n_items) if items is None else np.asarray(items)

    gmask     = np.stack([grid_masks(ds.estimates[:, p]) for p in range(ds.n_subs)])
    theta     = np.full((ds.n_subs, k + 1), np.nan)
    trial_nll = np.full((ds.n_subs, ds.n_items), np.nan)
    ok        = np.zeros(ds.n_subs, dtype=bool)
    failures  = []

    cues = ds.cues[idx]
    jobs = []
    for p in range(ds.n_subs):
        mask = ds.valid[idx, p]
        if mask.sum() == 0:
            failures.append(f"p{p}: no valid trials")
            continue
        jobs.append((p, k, ds.estimates[idx, p], mask, cues, ds.ex_cues, ds.ex_crit,
                     pis[p], ds.ub, float(np.nanstd(ds.estimates[:, p])) or 1.0,
                     dist, n_restarts, seed * 1000 + p, gmask[p][:, idx]))

    if n_jobs and n_jobs > 1 and source is not None:
        import multiprocessing as mp

        with mp.Pool(n_jobs, initializer=_worker_init,
                     initargs=(source, ds.cues, ds.ex_cues, ds.ex_crit)) as pool:
            results = pool.map(_worker_fit, jobs)
    else:
        _WORKER["model_fn"] = model_fn
        results = [_worker_fit(j) for j in jobs]

    for p, th, nll, err in results:
        if err is not None:
            failures.append(f"p{p}: {err}")
            continue
        theta[p]          = th
        trial_nll[p, idx] = nll
        ok[p]             = True

    msg = "" if not failures else f"{len(failures)} participant(s) failed: {failures[:3]}"
    return Fit(theta=theta, trial_nll=trial_nll, k=k + 1, ok=ok, message=msg)


def predictions(model_fn, fit, ds, p):
    """Model predictions for participant ``p`` on all test items."""
    return predict(model_fn, fit.theta[p][:-1], ds.cues, ds.ex_cues, ds.ex_crit)


def crossval_ll(model_fn, k, ds, pis=None, dist="normal", n_folds=5, n_restarts=3,
                seed=0, split_seed=0, n_jobs=1, source=None):
    """Held-out log-likelihood, k-fold over test items.

    AIC is easy for the LLM to game by adding parameters; this is the honest check on
    whether a discovered model actually generalises. Returns ``(total_ll, per_sub_ll)``
    where higher is better.

    ``split_seed`` fixes the item folds and must be held constant across the iterations of
    a run, otherwise re-scoring the *same* model lands hundreds of nats apart and the
    "did it improve?" test compares fold splits rather than models. ``seed`` varies only
    the optimiser restarts.
    """
    pis = granularity_weights(ds) if pis is None else pis

    order = np.random.default_rng(split_seed).permutation(ds.n_items)
    folds = np.array_split(order, n_folds)

    held  = np.full((ds.n_subs, ds.n_items), np.nan)
    for f in range(n_folds):
        test_idx  = folds[f]
        train_idx = np.setdiff1d(order, test_idx)
        fit = fit_all(model_fn, k, ds, pis=pis, dist=dist, n_restarts=n_restarts,
                      items=train_idx, seed=seed + f, n_jobs=n_jobs, source=source)
        for p in range(ds.n_subs):
            if not fit.ok[p]:
                continue
            mask = ds.valid[test_idx, p]
            if mask.sum() == 0:
                continue
            scale = float(np.nanstd(ds.estimates[:, p])) or 1.0
            try:
                nll = nll_reported(fit.theta[p], model_fn, ds.estimates[test_idx, p],
                                   ds.cues[test_idx], ds.ex_cues, ds.ex_crit, pis[p],
                                   ds.ub, sigma_scale=scale, dist=dist, agg="none",
                                   mask=mask)
            except ModelEvaluationError:
                continue
            held[p, test_idx] = -nll

    return float(np.nansum(held)), np.nansum(held, axis=1)
