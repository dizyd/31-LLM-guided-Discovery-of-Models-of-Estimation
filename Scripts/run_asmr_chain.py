#!/usr/bin/env python
"""
Command-line driver for the ASMR v2 chains -- the batch equivalent of sections 4 and 5 of
``ASMR_v2.ipynb``. This exists because a JupyterHub session is capped at four hours and the
full grid of chains does not fit into that.

A *chain* is one (seed model, run id) pair. The full grid is

    seeds  x  runs  =  4 x 5 = 20 chains,   each of ``--n-iterations`` steps.

Chains are completely independent, so they are handed out across the tasks of a Slurm job
array (``--task-id``/``--n-tasks``, both defaulted from the ``SLURM_ARRAY_*`` variables) in
a round-robin, which keeps every task's mix of seeds balanced.

Each chain writes into its own directory,

    ../Data/Model Outputs/asmr_v2/chains/<domain>_<seed>_run<id>/

so that concurrent array tasks never append to the same ``library.jsonl``. Merge them with
``merge_asmr_library.py`` before evaluating. A finished chain drops a ``DONE.json``; a
re-submitted or requeued job skips it, so resubmitting the same array is safe and picks up
where the last one stopped. Resumption is per chain, not per iteration: a chain killed
mid-flight restarts from its seed.

Examples
--------
    # everything, serially, on the current machine
    python run_asmr_chain.py

    # what one array task of 5 does (no LLM, just builds the first prompt)
    python run_asmr_chain.py --task-id 0 --n-tasks 5 --dry-run

    # one specific chain
    python run_asmr_chain.py --only GCM:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # asmr_data.py uses paths relative to Scripts/
sys.path.insert(0, str(HERE))

SEEDS = ("GCM", "CAM", "RulEx-J", "MAPP")


def build_chains(seeds, n_runs, only=None):
    """The full (seed, run_id) grid, or the explicit ``seed:run`` list given by --only."""
    if only:
        out = []
        for spec in only:
            seed, _, run_id = spec.partition(":")
            out.append((seed, int(run_id or 0)))
        return out
    return [(s, r) for s in seeds for r in range(n_runs)]


def env_int(name, default=None):
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="Mammals",
                   choices=["Mammals", "Food", "Countries"])
    p.add_argument("--seeds", nargs="+", default=list(SEEDS))
    p.add_argument("--n-runs", type=int, default=5,
                   help="independent replications per seed model")
    p.add_argument("--n-iterations", type=int, default=5,
                   help="propose-fit-critique steps within a chain")
    p.add_argument("--only", nargs="+", metavar="SEED:RUN",
                   help="run just these chains, e.g. --only GCM:0 MAPP:3")

    p.add_argument("--task-id", type=int, default=env_int("SLURM_ARRAY_TASK_ID"),
                   help="index of this array task (default: $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--n-tasks", type=int, default=env_int("SLURM_ARRAY_TASK_COUNT"),
                   help="size of the array (default: $SLURM_ARRAY_TASK_COUNT)")
    p.add_argument("--n-jobs", type=int, default=env_int("SLURM_CPUS_PER_TASK", 8),
                   help="processes for the per-participant fits "
                        "(default: $SLURM_CPUS_PER_TASK)")

    p.add_argument("--model", default="unsloth/Qwen3-32B-bnb-4bit")
    p.add_argument("--max-seq-length", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)

    p.add_argument("--n-restarts", type=int, default=5)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-restarts", type=int, default=3)

    p.add_argument("--out-root", default="../Data/Model Outputs/asmr_v2")
    p.add_argument("--force", action="store_true",
                   help="re-run chains that already have a DONE.json")
    p.add_argument("--dry-run", action="store_true",
                   help="no LLM: stop after building the first prompt of each chain")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="do not start a chain after this many seconds have elapsed; "
                        "the remaining chains are left for the next submission")
    return p.parse_args(argv)


def load_llm(args):
    """Load the reasoning model and wrap it in the ``generate`` callable."""
    from unsloth import FastLanguageModel

    from asmr_pipeline import make_generator

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(llm)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    return make_generator(llm, tokenizer, max_new_tokens=args.max_new_tokens), tokenizer


def main(argv=None):
    args    = parse_args(argv)
    t_start = time.time()

    chains = build_chains(args.seeds, args.n_runs, args.only)
    if args.task_id is not None and args.n_tasks:
        chains = chains[args.task_id::args.n_tasks]

    out_root = Path(args.out_root)
    print(f"host={os.environ.get('SLURMD_NODENAME', 'local')} "
          f"task={args.task_id}/{args.n_tasks} n_jobs={args.n_jobs}", flush=True)
    print(f"{len(chains)} chain(s) assigned: "
          f"{', '.join(f'{s}:{r}' for s, r in chains)}", flush=True)

    from asmr_data import load_aligned
    from asmr_pipeline import run

    ds = load_aligned(args.domain)
    print(f"loaded {args.domain}: {ds.n_subs} participants x {ds.n_items} test items",
          flush=True)

    def chain_dir(seed, run_id):
        return out_root / "chains" / f"{args.domain}_{seed}_run{run_id}"

    todo = [c for c in chains
            if args.force or not (chain_dir(*c) / "DONE.json").exists()]
    if not todo:
        print("nothing to do -- every assigned chain already has a DONE.json")
        return 0

    generate, tokenizer = None, None
    if not args.dry_run:
        generate, tokenizer = load_llm(args)

    n_done, n_failed = 0, 0
    for seed, run_id in chains:
        tag     = f"{args.domain}_{seed}_run{run_id}"
        out_dir = chain_dir(seed, run_id)
        done    = out_dir / "DONE.json"
        if done.exists() and not args.force:
            print(f"[{tag}] already done -- skipping", flush=True)
            continue
        if args.max_seconds is not None and time.time() - t_start > args.max_seconds:
            print(f"[{tag}] skipped: past the --max-seconds budget, "
                  f"resubmit to continue", flush=True)
            continue

        print(f"\n{'=' * 78}\n[{tag}] starting\n{'=' * 78}", flush=True)
        t0 = time.time()
        try:
            res = run(ds, seed=seed, run_id=run_id, n_iterations=args.n_iterations,
                      generate=generate, tokenizer=tokenizer,
                      n_jobs=args.n_jobs, n_restarts=args.n_restarts,
                      cv_folds=args.cv_folds, cv_restarts=args.cv_restarts,
                      max_new_tokens=args.max_new_tokens,
                      max_seq_length=args.max_seq_length,
                      temperature=args.temperature, residualize_format=True,
                      out_dir=out_dir)
        except Exception as exc:            # one bad chain must not take the whole job down
            n_failed += 1
            print(f"[{tag}] FAILED after {time.time() - t0:.0f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        n_done  += 1
        summary  = {"tag": tag, "domain": args.domain, "seed": seed, "run_id": run_id,
                    "n_iterations": args.n_iterations, "dry_run": args.dry_run,
                    "best_cv_ll": res["best_cv_ll"], "n_models": len(res["library"]),
                    "seconds": time.time() - t0,
                    "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
        (out_dir / "best_model.py").write_text(res["best_source"] or "", encoding="utf-8")
        if not args.dry_run:
            done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{tag}] done in {summary['seconds'] / 60:.1f} min -- "
              f"best held-out LL {res['best_cv_ll']:.1f}, "
              f"{summary['n_models']} models into the library", flush=True)

    print(f"\n{n_done} chain(s) finished, {n_failed} failed, "
          f"{(time.time() - t_start) / 3600:.2f} h wall", flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
