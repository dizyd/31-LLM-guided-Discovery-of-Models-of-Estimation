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

``--adopt-flat`` first pulls in chains from the *old* flat layout, i.e. anything run from
the notebook or from a JupyterHub session before ``run_asmr_chain.py`` existed, which wrote
``asmr_v2/<tag>_iter<n>.npz`` and one shared ``asmr_v2/library.jsonl``. It copies (never
moves) those into the per-chain layout and marks a chain done once it has all of its
iterations, so a resubmitted array skips it:

    python merge_asmr_library.py --adopt-flat --n-iterations 5
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

_FLAT = re.compile(r"^(?P<tag>.+)_iter(?P<iter>\d+)\.npz$")


def adopt_flat(out_root, n_iterations):
    """Re-file results from the pre-array flat layout into ``chains/<tag>/``."""
    npzs = defaultdict(dict)
    for path in sorted(out_root.glob("*_iter*.npz")):
        m = _FLAT.match(path.name)
        if m:
            npzs[m["tag"]][int(m["iter"])] = path
    if not npzs:
        print("--adopt-flat: no flat <tag>_iter<n>.npz files to adopt")
        return

    flat_lib = out_root / "library.jsonl"
    by_tag   = defaultdict(list)
    if flat_lib.exists():
        for line in flat_lib.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                by_tag[entry.get("tag", "?")].append(entry)
        # merge() overwrites library.jsonl from the shards, so keep the original.
        shutil.copy2(flat_lib, out_root / "library.jsonl.flat.bak")
        print(f"--adopt-flat: backed up {flat_lib.name} -> library.jsonl.flat.bak")

    for tag, iters in sorted(npzs.items()):
        chain_dir = out_root / "chains" / tag
        chain_dir.mkdir(parents=True, exist_ok=True)
        for path in iters.values():
            target = chain_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)

        shard = chain_dir / "library.jsonl"
        if by_tag.get(tag) and not shard.exists():
            with open(shard, "w", encoding="utf-8") as fh:
                for entry in by_tag[tag]:
                    fh.write(json.dumps(entry) + "\n")

        complete = set(iters) >= set(range(n_iterations))
        done     = chain_dir / "DONE.json"
        if complete and not done.exists():
            done.write_text(json.dumps(
                {"tag": tag, "adopted_from": "flat layout",
                 "n_iterations": n_iterations,
                 "n_models": len(by_tag.get(tag, []))}, indent=2), encoding="utf-8")
        print(f"  {tag}: {len(iters)} iteration(s), "
              f"{len(by_tag.get(tag, []))} library entries"
              f"{' -- marked done' if complete else ' -- incomplete, will be re-run'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="../Data/Model Outputs/asmr_v2")
    p.add_argument("--domain", default=None,
                   help="keep only entries from this domain")
    p.add_argument("--adopt-flat", action="store_true",
                   help="first re-file results written in the old flat layout")
    p.add_argument("--n-iterations", type=int, default=5,
                   help="iterations a chain needs before --adopt-flat calls it done")
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    if args.adopt_flat:
        adopt_flat(out_root, args.n_iterations)

    shards = sorted((out_root / "chains").glob("*/library.jsonl"))
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
