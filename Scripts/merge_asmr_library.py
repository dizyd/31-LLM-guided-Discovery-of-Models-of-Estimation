#!/usr/bin/env python
"""
Merge the per-chain ``library.jsonl`` files written by ``run_asmr_chain.py`` into the single
``asmr_v2/library.jsonl`` that ``asmr_evaluate.load_library`` reads.

Array tasks run concurrently, so each chain appends to its own file
(``asmr_v2/chains/<tag>/library.jsonl``) rather than to a shared one. This concatenates
them, de-duplicating by ``model_id`` -- two chains that converge on the same model produce
the same hash, and the library is a set of model *forms*, so the duplicate carries no extra
information. Where a model id appears more than once the entry with the best held-out
log-likelihood is kept, so the recorded provenance points at the chain that fitted it best.

    python merge_asmr_library.py                 # all domains
    python merge_asmr_library.py --domain Mammals
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="../Data/Model Outputs/asmr_v2")
    p.add_argument("--domain", default=None,
                   help="keep only entries from this domain")
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    shards   = sorted((out_root / "chains").glob("*/library.jsonl"))
    if not shards:
        print(f"no per-chain library.jsonl under {out_root / 'chains'}")
        return 1

    best, n_lines = {}, 0
    for shard in shards:
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                n_lines += 1
                if args.domain is not None and entry.get("domain") != args.domain:
                    continue
                mid  = entry["model_id"]
                prev = best.get(mid)
                if prev is None or _score(entry) > _score(prev):
                    best[mid] = entry

    target = out_root / "library.jsonl"
    with open(target, "w", encoding="utf-8") as fh:
        for entry in best.values():
            fh.write(json.dumps(entry) + "\n")

    by_seed = {}
    for e in best.values():
        by_seed[e.get("seed", "?")] = by_seed.get(e.get("seed", "?"), 0) + 1
    print(f"{len(shards)} chain(s), {n_lines} entries -> "
          f"{len(best)} unique models in {target}")
    for seed, n in sorted(by_seed.items()):
        print(f"  from seed {seed:>8}: {n}")
    return 0


def _score(entry):
    val = entry.get("cv_ll")
    return float("-inf") if val is None else float(val)


if __name__ == "__main__":
    sys.exit(main())
