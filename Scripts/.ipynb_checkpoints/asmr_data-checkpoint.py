"""
Alignment-safe data loading for the ASMR pipeline.

The v1 pipeline (``ASMR.ipynb``) silently mis-indexed both participants and trials:

* ``nll_centaur`` rows are in ``ID_n`` order (``narrative_data.csv``, sorted by the
  Prolific ID string), while ``exp_data.iloc[:, participant]`` used the *column order*
  of ``data_analysis_<domain>.csv``. Those are the same set in a different order.
* ``extract_nll_testing_phase_only.py`` sorts each participant's NLLs by item ID, while
  ``until_nth_occurrence(text, "<<", ...)`` walks the narrative in *presentation* order.

Everything here is keyed by **item ID** and by the ``ID_n`` participant index, and a hard
assertion at load time re-derives the estimates from the narrative text and checks them
against the behavioural CSV so the alignment can never silently break again.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from configurations import DOMAIN_CONFIG

NARRATIVE_PATH = "../Data/Preprocessed Data/narrative_data.csv"
TIDY_PATH      = "../Data/Behavioral Data/data_tidy_combined.csv"
CENTAUR_PATH   = "../Data/Model Outputs/log_likelihood_{domain}_TESTING_ALIGNED.pth"


@dataclass
class Dataset:
    """Everything the pipeline needs, all consistently ordered.

    Row order of every ``(n_items, ...)`` array is ascending item ID; the participant
    axis of every ``(n_subs, ...)`` array is the ``ID_n`` participant index.
    """

    domain:      str
    n_dim:       int
    ub:          float

    # --- test items (ascending item ID) ---
    item_ids:    np.ndarray          # (n_items,) int
    item_names:  np.ndarray          # (n_items,) str
    cues:        np.ndarray          # (n_items, n_dim)
    true_crit:   np.ndarray          # (n_items,)

    # --- exemplars ---
    ex_ids:      np.ndarray          # (n_ex,) int
    ex_names:    np.ndarray          # (n_ex,) str
    ex_cues:     np.ndarray          # (n_ex, n_dim)
    ex_crit:     np.ndarray          # (n_ex,)

    # --- participants (ID_n order) ---
    participant_ids: list            # (n_subs,) Prolific IDs, index == ID_n
    estimates:   np.ndarray          # (n_items, n_subs) test-phase estimates
    centaur_nll: np.ndarray          # (n_subs, n_items) nats, probability *mass*
    pres_pos:    np.ndarray          # (n_items, n_subs) int, presentation position
    train_est:   list                # (n_subs,) 1-D arrays of training-phase estimates
    valid:       np.ndarray          # (n_items, n_subs) bool, False for excluded trials

    @property
    def n_items(self):
        return len(self.item_ids)

    @property
    def n_subs(self):
        return len(self.participant_ids)

    def long_frame(self):
        """Tidy long frame, one row per participant x test item."""
        n_i, n_s = self.n_items, self.n_subs
        out = pd.DataFrame({
            "participant_idx":  np.repeat(np.arange(n_s), n_i),
            "participant_id":   np.repeat(np.asarray(self.participant_ids), n_i),
            "item_id":          np.tile(self.item_ids, n_s),
            "item":             np.tile(self.item_names, n_s),
            "true_crit":        np.tile(self.true_crit, n_s),
            "estimate":         self.estimates.T.reshape(-1),
            "presentation_pos": self.pres_pos.T.reshape(-1),
            "centaur_nll":      self.centaur_nll.reshape(-1),
            "valid":            self.valid.T.reshape(-1),
        })
        for d in range(self.n_dim):
            out[f"cue_{d + 1}"] = np.tile(self.cues[:, d], n_s)
        return out


def _narrative(domain):
    nd = pd.read_csv(NARRATIVE_PATH)
    nd = nd[nd["domain"] == domain].sort_values("participant").reset_index(drop=True)
    assert (nd["participant"].to_numpy() == np.arange(len(nd))).all(), \
        "narrative_data participant index is not a dense 0..n-1 range"
    return nd


def _training_estimates(domain, participant_ids):
    """Per-participant training-phase estimates, in ID_n order.

    Used to fit the response-granularity weights out of sample, so they cost no free
    parameters in the test-phase fit.
    """
    tidy = pd.read_csv(TIDY_PATH, sep=";", decimal=",")
    tidy = tidy[(tidy["domain"] == domain) & (tidy["phase"] == "training")]
    by_id = {pid: np.asarray(g["est"], dtype=float) for pid, g in tidy.groupby("ID")}
    out = []
    for pid in participant_ids:
        e = by_id.get(pid, np.array([], dtype=float))
        out.append(e[np.isfinite(e)])
    return out


def load_aligned(domain, check=True):
    """Load ``domain`` with participants and trials guaranteed to line up.

    Set ``check=False`` only to skip the (cheap) narrative cross-check.
    """
    cfg      = DOMAIN_CONFIG[domain]
    n_dim    = cfg["n_dim"]
    cue_cols = [f"V{i}" for i in range(1, n_dim + 1)]

    design = pd.read_csv(cfg["stim"], sep=";", decimal=",")
    train  = design.loc[design["training"] == 1]
    test   = design.loc[design["training"] == 0].sort_values("ID")

    nd              = _narrative(domain)
    participant_ids = nd["ID"].tolist()

    behav = pd.read_csv(cfg["data"], sep=",")
    behav = behav[behav["training"] == 0].sort_values("ID_item")
    assert (behav["ID_item"].to_numpy() == test["ID"].to_numpy()).all(), \
        "design and behavioural files disagree on the set of test items"
    missing = set(participant_ids) - set(behav.columns)
    assert not missing, f"participants missing from the behavioural file: {sorted(missing)}"

    # Reindex by ID_n order -- do NOT rely on the CSV column order.
    estimates = behav[participant_ids].to_numpy(dtype=float)

    centaur = torch.load(CENTAUR_PATH.format(domain=domain), weights_only=True)
    centaur = torch.stack(centaur).float().numpy()
    assert centaur.shape == (len(participant_ids), len(test)), (
        f"Centaur NLL shape {centaur.shape} != "
        f"({len(participant_ids)}, {len(test)}); re-run extract_nll_testing_phase_only.py"
    )

    # Presentation position of every test item, per participant.
    item_ids = test["ID"].to_numpy(dtype=int)
    rank     = {int(i): k for k, i in enumerate(item_ids)}
    pres_pos = np.full((len(item_ids), len(participant_ids)), -1, dtype=int)
    for p, row in nd.iterrows():
        order = [int(x) for x in str(row["ID_items"]).split(",")][-len(item_ids):]
        for pos, iid in enumerate(order):
            pres_pos[rank[iid], p] = pos
    assert (pres_pos >= 0).all(), "some test item never appeared in ID_items"

    ds = Dataset(
        domain          = domain,
        n_dim           = n_dim,
        ub              = float(cfg["ub"]),
        item_ids        = item_ids,
        item_names      = test["item"].to_numpy(),
        cues            = test[cue_cols].to_numpy(dtype=float),
        true_crit       = test["crit"].to_numpy(dtype=float),
        ex_ids          = train["ID"].to_numpy(dtype=int),
        ex_names        = train["item"].to_numpy(),
        ex_cues         = train[cue_cols].to_numpy(dtype=float),
        ex_crit         = train["crit"].to_numpy(dtype=float),
        participant_ids = participant_ids,
        estimates       = estimates,
        centaur_nll     = centaur,
        pres_pos        = pres_pos,
        train_est       = _training_estimates(domain, participant_ids),
        valid           = np.isfinite(estimates),
    )

    if check:
        assert_alignment(ds, nd)

    n_drop = int((~ds.valid).sum())
    print(f"[{domain}] {ds.n_subs} participants x {ds.n_items} test items, "
          f"{len(ds.ex_ids)} exemplars, {n_dim} cues -- alignment verified"
          + (f" ({n_drop} excluded trials masked)" if n_drop else ""))
    return ds


def assert_alignment(ds, nd=None):
    """Re-derive the test estimates from the narrative text and compare.

    This is the guard against the v1 bugs: it fails loudly if either the participant
    order or the item order drifts apart again.
    """
    nd = _narrative(ds.domain) if nd is None else nd
    rank = {int(i): k for k, i in enumerate(ds.item_ids)}

    for p, row in nd.iterrows():
        assert row["ID"] == ds.participant_ids[p]
        order = [int(x) for x in str(row["ID_items"]).split(",")][-ds.n_items:]
        said  = re.findall(r"<<(.*?)>>", row["text"])[-ds.n_items:]
        assert len(said) == ds.n_items, (
            f"participant {p}: found {len(said)} test responses in the narrative, "
            f"expected {ds.n_items}"
        )
        from_text = np.empty(ds.n_items, dtype=float)
        for iid, val in zip(order, said):
            from_text[rank[iid]] = float(val)

        obs  = ds.estimates[:, p]
        seen = np.isfinite(obs)

        # Trials the manuscript excluded (estimate above the domain ceiling) are NaN in
        # the behavioural file but still present in the narrative Centaur was scored on.
        # Accept those, but only if the narrative value really is out of range.
        for i in np.flatnonzero(~seen):
            if not from_text[i] > ds.ub:
                raise AssertionError(
                    f"participant {p} ({row['ID']}): item {ds.item_ids[i]} is missing "
                    f"from the behavioural file but the narrative value {from_text[i]} "
                    f"is within range [0, {ds.ub}] -- this is not a manuscript exclusion"
                )

        if not np.allclose(from_text[seen], obs[seen]):
            bad = np.flatnonzero(seen)[~np.isclose(from_text[seen], obs[seen])][:5]
            raise AssertionError(
                f"ALIGNMENT BROKEN for participant {p} ({row['ID']}): "
                f"narrative says {from_text[bad]} but the behavioural file says "
                f"{obs[bad]} for items {ds.item_ids[bad]}"
            )
    return True


if __name__ == "__main__":
    for dom in ("Mammals", "Food", "Countries"):
        try:
            load_aligned(dom)
        except Exception as exc:          # noqa: BLE001 - diagnostic entry point
            print(f"[{dom}] {type(exc).__name__}: {exc}")
