#!/usr/bin/env python
"""
Batch equivalent of sections 6 and 7 of ``ASMR_v2.ipynb``: score every library member on
every participant with held-out items, assign each participant their best member, and write
the tables out.

No GPU is involved -- this is scipy optimisation only -- so it belongs on a ``cpu`` node,
where the whole 96-core node can go into ``--n-jobs``. Run ``merge_asmr_library.py`` first.

Writes into ``../Data/Model Outputs/asmr_v2/``:

    scores_<domain>.csv     one row per (library member, participant), held-out LL
    winners_<domain>.csv    the best member per participant
    counts_<domain>.csv     participants per winning model (cf. Manuscript Table 2)
    reference_<domain>.csv  Centaur and the response-style nulls to read them against
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))


def env_int(name, default=None):
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="Mammals",
                   choices=["Mammals", "Food", "Countries"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=env_int("SLURM_CPUS_PER_TASK", 8))
    p.add_argument("--out-root", default="../Data/Model Outputs/asmr_v2")
    args = p.parse_args(argv)

    import numpy as np
    np.seterr(over="ignore", invalid="ignore")

    from asmr_data import load_aligned
    from asmr_evaluate import (evaluate_library, load_library, reference_rows,
                               response_style_summary, winner_table)

    out_root = Path(args.out_root)
    ds       = load_aligned(args.domain)
    library  = load_library(out_dir=out_root, domain=args.domain)
    print(f"{args.domain}: {ds.n_subs} participants, "
          f"{len(library)} models in the library (incl. seeds and references)", flush=True)

    print("\n--- response style ---")
    print(response_style_summary(ds).to_string(index=False))
    print("\n--- references ---")
    refs = reference_rows(ds)
    print(refs.to_string(index=False))

    print("\n--- scoring the library ---", flush=True)
    scores       = evaluate_library(ds, library, n_folds=args.n_folds,
                                    n_restarts=args.n_restarts, n_jobs=args.n_jobs)
    best, counts = winner_table(scores)

    print("\n--- participants per winning model ---")
    print(counts.to_string(index=False))

    scores.to_csv(out_root / f"scores_{args.domain}.csv", index=False)
    best.to_csv(out_root / f"winners_{args.domain}.csv", index=False)
    counts.to_csv(out_root / f"counts_{args.domain}.csv", index=False)
    refs.to_csv(out_root / f"reference_{args.domain}.csv", index=False)
    print(f"\nwritten to {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
