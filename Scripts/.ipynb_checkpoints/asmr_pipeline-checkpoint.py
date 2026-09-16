"""
The ASMR loop: fit -> regret -> prompt -> generate -> validate -> iterate.

Control-flow fixes over v1 (``ASMR.ipynb``):

* the rollback is on again (it was commented out in the per-participant loop), and it no
  longer ``continue``s past the SRM step -- in v1 a rollback silently burned an iteration
  re-fitting a model identical to the previous one, because the prompt was never built;
* the model that is kept is the **best** one seen, by held-out log-likelihood, not the last
  one generated;
* an invalid generation is caught by the validator, retried once with the error message
  appended, and only then rolled back -- instead of becoming a ``1e12`` sentinel NLL;
* ``temperature`` always has a value (v1 raised ``NameError`` if iteration 0 threw before
  ``temperature_setting`` was assigned);
* the prompt is length-checked against the context window before generation.

The LLM is injected as a ``generate(prompt) -> str`` callable, so this module does not
depend on unsloth or transformers and can be exercised without a GPU.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import asmr_srm as srm
from asmr_codegen import retry_prompt, strip_llm_wrapping, validate_model_code
from asmr_fit import crossval_ll, fit_all, granularity_weights, predictions
from asmr_models_seed import SEED_MODELS

OUT_DIR = Path("../Data/Model Outputs/asmr_v2")


def model_hash(source):
    """Stable id for a model, insensitive to whitespace-only edits."""
    norm = "\n".join(l.rstrip() for l in source.strip().splitlines() if l.strip())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


@dataclass
class Iteration:
    iteration:  int
    source:     str
    model_id:   str
    valid:      bool
    k:          int          = 0
    aic:        float        = np.nan
    nats_per_trial: float    = np.nan
    cv_ll:      float        = None
    n_ok:       int          = 0
    n_points:   int          = 0
    prompt_tokens: int       = 0
    error:      str          = ""
    warnings:   list         = field(default_factory=list)
    rolled_back: bool        = False
    seconds:    float        = 0.0


def _record_dict(it):
    d = asdict(it)
    d["source"] = it.source
    return d


def run(ds, seed="GCM", n_iterations=5, generate=None, tokenizer=None, run_id=0,
        top_k_per_sub=8, max_points=60, dist="normal", n_restarts=5, cv_folds=5,
        cv_restarts=4, n_jobs=1, residualize_format=True,
        max_new_tokens=8192, max_seq_length=40960,
        temperature=0.7, out_dir=OUT_DIR, verbose=True):
    """Run one ASMR chain.

    ``generate`` is called as ``generate(prompt, temperature=...)`` and must return the
    raw model output. Pass ``None`` for a dry run that stops after the first prompt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{ds.domain}_{seed}_run{run_id}"

    pis     = granularity_weights(ds)
    source  = SEED_MODELS[seed] if seed in SEED_MODELS else seed
    records, history, library = [], [], {}

    best_source, best_score, best_fit, best_preds = None, -np.inf, None, None

    for it in range(n_iterations):
        t0  = time.time()
        rec = Iteration(iteration=it, source=source, model_id=model_hash(source),
                        valid=False)

        report = validate_model_code(source, ds.cues, ds.ex_cues, ds.ex_crit)
        if not report.ok:
            rec.error = report.error
            if best_source is None:
                records.append(rec)
                raise RuntimeError(
                    f"the seed model itself failed validation:\n{report.error}"
                )
            if verbose:
                print(f"[{tag}] iter {it}: candidate rejected -- {report.error.splitlines()[0]}")
            source, rec.rolled_back = best_source, True
            report = validate_model_code(source, ds.cues, ds.ex_cues, ds.ex_crit)
            rec.source, rec.model_id = source, model_hash(source)

        rec.valid    = True
        rec.warnings = report.warnings
        rec.k        = report.num_parameters

        fit = fit_all(report.model_fn, report.num_parameters, ds, pis=pis, dist=dist,
                      n_restarts=n_restarts, seed=run_id * 100 + it,
                      n_jobs=n_jobs, source=report.source)
        if not fit.ok.any():
            rec.error = "no participant could be fitted: " + fit.message
            records.append(rec)
            if best_source is None:
                raise RuntimeError(rec.error)
            source, rec.rolled_back = best_source, True
            continue

        rec.aic            = fit.aic
        rec.nats_per_trial = fit.nats_per_trial
        rec.n_ok           = int(fit.ok.sum())

        cv, _   = crossval_ll(report.model_fn, report.num_parameters, ds, pis=pis,
                              dist=dist, n_folds=cv_folds, n_restarts=cv_restarts,
                              seed=run_id * 100 + it, split_seed=run_id,
                              n_jobs=n_jobs, source=report.source)
        rec.cv_ll = cv

        preds = {p: (predictions(report.model_fn, fit, ds, p) if fit.ok[p]
                     else np.full(ds.n_items, np.nan))
                 for p in range(ds.n_subs)}

        library[rec.model_id] = {"source": source, "k": rec.k, "cv_ll": cv,
                                 "aic": rec.aic, "iteration": it, "seed": seed,
                                 "run_id": run_id, "domain": ds.domain}

        improved = cv > best_score
        if improved:
            best_source, best_score, best_fit, best_preds = source, cv, fit, preds
        if verbose:
            print(f"[{tag}] iter {it}: k={rec.k:2d} AIC={rec.aic:9.1f} "
                  f"nats/trial={rec.nats_per_trial:.3f} heldout LL={cv:9.1f} "
                  f"{'** best' if improved else '(kept best)'}")

        history.append({"iteration": it, "k": rec.k, "aic": rec.aic, "cv_ll": cv})

        # Always build the prompt from the *best* model so far -- v1's rollback skipped
        # the prompt entirely and re-fitted the same model on the next pass.
        prompt_source = best_source
        prompt_fit    = best_fit
        prompt_preds  = best_preds

        delta = srm.regret(prompt_fit, ds)
        picks = srm.select_points(delta, top_k_per_sub=top_k_per_sub,
                                  max_points=max_points, ds=ds,
                                  residualize_format=residualize_format,
                                  rng=np.random.default_rng(run_id * 100 + it))
        rec.n_points = len(picks)

        prompt = srm.compose(prompt_source, ds, prompt_fit, prompt_preds, picks, history)
        if tokenizer is not None:
            rec.prompt_tokens = srm.check_prompt_fits(prompt, tokenizer, max_new_tokens,
                                                      max_seq_length)

        rec.seconds = time.time() - t0
        records.append(rec)
        _save(out_dir, tag, it, rec, fit, delta, picks, prompt, "")

        if generate is None or it == n_iterations - 1:
            if generate is None and verbose:
                print(f"[{tag}] dry run: stopping after building the first prompt "
                      f"({rec.n_points} SRM points, {rec.prompt_tokens or '?'} tokens)")
            break

        raw       = generate(prompt, temperature=temperature)
        candidate = strip_llm_wrapping(raw)
        check     = validate_model_code(candidate, ds.cues, ds.ex_cues, ds.ex_crit)
        if not check.ok:
            if verbose:
                print(f"[{tag}] iter {it}: retrying after validator error -- "
                      f"{check.error.splitlines()[0]}")
            raw       = generate(retry_prompt(prompt, check), temperature=temperature)
            candidate = strip_llm_wrapping(raw)

        _save(out_dir, tag, it, rec, fit, delta, picks, prompt, raw)
        source = candidate

    _write_library(out_dir, tag, library)
    return {"records": records, "library": library, "best_source": best_source,
            "best_cv_ll": best_score, "best_fit": best_fit, "tag": tag}


def _save(out_dir, tag, it, rec, fit, delta, picks, prompt, raw):
    np.savez(out_dir / f"{tag}_iter{it}.npz",
             trial_nll=fit.trial_nll, theta=fit.theta, ok=fit.ok, delta=delta,
             picks=np.array(picks, dtype=object), prompt=prompt, raw_output=raw,
             record=json.dumps(_record_dict(rec)))


def _write_library(out_dir, tag, library):
    path = out_dir / "library.jsonl"
    seen = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            seen = {json.loads(l)["model_id"] for l in fh if l.strip()}
    with open(path, "a", encoding="utf-8") as fh:
        for mid, entry in library.items():
            if mid in seen:
                continue
            fh.write(json.dumps({"model_id": mid, "tag": tag, **entry}) + "\n")


def make_generator(llm, tokenizer, max_new_tokens=8192):
    """Wrap an unsloth/transformers pipeline into the ``generate`` callable."""
    from transformers import pipeline

    gen = pipeline("text-generation", model=llm, tokenizer=tokenizer,
                   device_map="auto", return_full_text=False)

    def generate(prompt, temperature=0.7):
        out = gen([{"role": "system", "content": ""},
                   {"role": "user", "content": prompt}],
                  do_sample=True, temperature=temperature, top_p=0.95, top_k=20,
                  min_p=0.0, return_full_text=False, max_new_tokens=max_new_tokens)
        return out[0]["generated_text"]

    return generate


if __name__ == "__main__":
    from asmr_data import load_aligned

    ds  = load_aligned("Mammals")
    res = run(ds, seed="GCM", n_iterations=1, generate=None, n_restarts=3,
              cv_folds=3, cv_restarts=2)
    npz = np.load(OUT_DIR / f"{res['tag']}_iter0.npz", allow_pickle=True)
    print("\n" + "=" * 78 + "\nPROMPT PREVIEW\n" + "=" * 78)
    print(str(npz["prompt"]))
