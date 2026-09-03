"""
Parsing and validation of LLM-authored model code.

In v1 a broken generation was never detected: ``nll_truncnorm_woS`` returned ``1e12`` per
trial on any exception, so a model that crashed produced ``nll.sum() = 6.8e13`` (see
``srm_model_1p_0_iteration_4.npz``), the ``try`` block never fired, and the rollback -- which
was commented out anyway -- had nothing to react to.

Every candidate now goes through :func:`validate_model_code` before it is allowed anywhere
near the optimiser. A candidate that fails is reported, not silently scored.
"""

import ast
import re
import traceback
from dataclasses import dataclass, field

import numpy as np

ALLOWED_IMPORTS = {"numpy", "np", "math", "scipy", "scipy.special", "scipy.stats"}
REQUIRED_SIG    = ("parameters", "cues", "ex_cues", "ex_crit")
MAX_PARAMETERS  = 60


@dataclass
class ValidationReport:
    ok:             bool
    source:         str
    num_parameters: int  = 0
    model_fn:       object = None
    error:          str  = ""
    warnings:       list = field(default_factory=list)


def strip_llm_wrapping(text):
    """Remove reasoning traces and markdown fences from a raw generation."""
    text = text.split("</think>")[-1]
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if fences:
        text = max(fences, key=len)
    return text.strip()


def _indexed_parameter_slots(tree):
    """Which ``parameters[...]`` slots are read, as far as static analysis can tell.

    Returns ``(explicit_indices, uses_slice)``. A slice is treated as "covers whatever it
    covers" -- we only use this to catch models that leave whole parameters dead.
    """
    idx, sliced = set(), False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "parameters"):
            continue
        s = node.slice
        if isinstance(s, ast.Constant) and isinstance(s.value, int):
            idx.add(s.value)
        elif isinstance(s, ast.UnaryOp) and isinstance(s.op, ast.USub) \
                and isinstance(s.operand, ast.Constant):
            idx.add(-s.operand.value)
        else:
            sliced = True
    return idx, sliced


def validate_model_code(source, cues, ex_cues, ex_crit, n_draws=20, rng=None):
    """Statically and dynamically check a candidate model function.

    Checks, in order: syntax; no unexpected imports; ``NUM_PARAMETERS`` present and sane;
    a top-level ``def model`` with the exact expected signature; executes; returns a
    finite ``(n_trials,)`` array for every one of ``n_draws`` random parameter vectors;
    and actually *responds* to its parameters.
    """
    rng    = np.random.default_rng() if rng is None else rng
    source = strip_llm_wrapping(source)
    rep    = ValidationReport(ok=False, source=source)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        rep.error = f"SyntaxError: {exc}"
        return rep

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        bad = names - ALLOWED_IMPORTS
        if bad:
            rep.error = f"disallowed import(s): {sorted(bad)}"
            return rep

    fn_node = next((n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "model"), None)
    if fn_node is None:
        rep.error = "no top-level 'def model(...)' found"
        return rep

    args = [a.arg for a in fn_node.args.args]
    if tuple(args[:4]) != REQUIRED_SIG:
        rep.error = (f"signature must start with {REQUIRED_SIG}, got {tuple(args)}")
        return rep

    ns = {"np": np, "numpy": np}
    try:
        exec(compile(tree, "<llm_model>", "exec"), ns)          # noqa: S102
    except Exception:                                           # noqa: BLE001
        rep.error = "error while executing the module:\n" + traceback.format_exc(limit=3)
        return rep

    if "NUM_PARAMETERS" not in ns:
        rep.error = "NUM_PARAMETERS is not defined"
        return rep
    k = ns["NUM_PARAMETERS"]
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= MAX_PARAMETERS:
        rep.error = f"NUM_PARAMETERS must be an int in [1, {MAX_PARAMETERS}], got {k!r}"
        return rep
    rep.num_parameters = int(k)

    model_fn = ns["model"]

    # -- dynamic smoke test -------------------------------------------------------
    outs = []
    for scale in (1.0, 3.0):
        for _ in range(n_draws // 2):
            theta = rng.normal(0.0, scale, size=rep.num_parameters)
            try:
                pred = model_fn(theta, cues, ex_cues, ex_crit)
            except Exception:                                   # noqa: BLE001
                rep.error = (f"raised for parameters ~ N(0, {scale}):\n"
                             + traceback.format_exc(limit=3))
                return rep
            try:
                pred = np.asarray(pred, dtype=float)
            except Exception:                               # noqa: BLE001
                rep.error = ("returned something that is not a numeric array -- return "
                             "only pred_crit, not a tuple such as (pred_crit, sigma)")
                return rep
            if pred.shape != (cues.shape[0],):
                rep.error = (f"returned shape {pred.shape}, expected "
                             f"{(cues.shape[0],)} -- return only pred_crit, not a tuple")
                return rep
            if not np.all(np.isfinite(pred)):
                rep.error = f"returned NaN/inf for parameters ~ N(0, {scale})"
                return rep
            outs.append(pred)

    outs = np.asarray(outs)
    if np.allclose(outs, outs[0]):
        rep.error = "predictions do not depend on the parameters at all"
        return rep

    # -- dead parameters ----------------------------------------------------------
    idx, sliced = _indexed_parameter_slots(tree)
    if not sliced:
        dead = sorted(set(range(rep.num_parameters)) - {i % rep.num_parameters
                                                        for i in idx})
        if dead:
            rep.error = (f"NUM_PARAMETERS = {rep.num_parameters} but parameter(s) {dead} "
                         f"are never used")
            return rep

    # A parameter can still be dead behind a slice; flag it as a warning rather than a
    # hard failure, since AIC will punish it anyway.
    base = model_fn(np.zeros(rep.num_parameters), cues, ex_cues, ex_crit)
    inert = []
    for i in range(rep.num_parameters):
        t = np.zeros(rep.num_parameters)
        t[i] = 1.0
        try:
            if np.allclose(model_fn(t, cues, ex_cues, ex_crit), base):
                inert.append(i)
        except Exception:                                       # noqa: BLE001
            inert.append(i)
    if inert:
        rep.warnings.append(f"parameter(s) {inert} do not change the prediction at theta=0")

    rep.model_fn = model_fn
    rep.ok       = True
    return rep


def retry_prompt(prompt, report):
    """Append a validator failure to the original prompt so the LLM can repair it."""
    return (
        prompt
        + "\n\nYour previous answer was rejected by the automatic validator with this "
          "error:\n\n"
        + report.error.strip()
        + "\n\nHere is the code that was rejected:\n\n"
        + report.source
        + "\n\nPlease fix this and output the corrected model, following the same rules. "
          "Output ONLY the code."
    )


if __name__ == "__main__":
    from asmr_data import load_aligned
    from asmr_models_seed import SEED_MODELS

    ds = load_aligned("Mammals")

    print("\n--- seeds ---")
    for name, src in SEED_MODELS.items():
        rep = validate_model_code(src, ds.cues, ds.ex_cues, ds.ex_crit)
        print(f"{name:>8}: {'OK  ' if rep.ok else 'FAIL'} k={rep.num_parameters} "
              f"{rep.error.splitlines()[0] if rep.error else ''}")
        for w in rep.warnings:
            print(f"{'':>10}warning: {w}")

    print("\n--- v1 generations that produced sentinel NLLs ---")
    import glob

    for path in sorted(glob.glob("../Data/Model Outputs/srm_model_*.npz")):
        z = np.load(path, allow_pickle=True)
        if float(np.asarray(z["nll"]).sum()) < 1e9:
            continue
        rep = validate_model_code(str(z["model_string"]),
                                  ds.cues, ds.ex_cues, ds.ex_crit)
        first = rep.error.splitlines()[0] if rep.error else ""
        print(f"{path.split('/')[-1]:>40}: "
              f"{'ACCEPTED (!)' if rep.ok else 'rejected'}  {first[:70]}")
