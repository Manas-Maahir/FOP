#!/usr/bin/env python
"""Run the paper's Table 8 ablation locally, over multiple seeds, and aggregate the result.

[report.md](report.md) §6 could not separate the ablation conditions: the spread across six
*different* configurations was ~2.3 AP50, while two runs of the *same* configuration differed by
~0.8. With one seed per cell, that is an unanswerable question -- and §7 named multi-seed averaging
as the single highest-impact fix. On Colab it was unaffordable. Locally it is just time, so seeds
are a first-class argument here and the output is **mean +/- std**, not a single number.

    python tools/ablate.py --data-root data/tbx11k_512                     # 13 cells, seed 0
    python tools/ablate.py --data-root data/tbx11k_512 --seeds 0 1 2       # the real experiment
    python tools/ablate.py --data-root data/tbx11k_512 --cells none_none symattention_spe_stn_r2l

Every cell is skipped if its run directory already holds a finished result, so an interrupted sweep
resumes by re-running the same command. Budget roughly 10-15 min per cell per seed on an 8 GB
laptop GPU: 13 cells x 3 seeds is an overnight job.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Paper Table 8: attention x positional encoding x symmetry direction.
# Ordered most-informative first, so a sweep that is cut short is still worth reading:
# the baseline and the full model come out before the filler cells.
CELLS: list[tuple[str, list[str]]] = [
    ("none_none", ["--no-sas"]),                                                    # baseline row
    ("symattention_spe_stn_r2l", ["--attention", "symattention", "--pe", "spe",
                                  "--stn", "--direction", "r2l"]),                  # full SymFormer
    ("vanilla_ape", ["--attention", "vanilla", "--pe", "ape"]),
    ("symattention_ape", ["--attention", "symattention", "--pe", "ape"]),
    ("symattention_spe_nostn_r2l", ["--attention", "symattention", "--pe", "spe",
                                    "--no-stn", "--direction", "r2l"]),
    ("symattention_spe_stn_l2r", ["--attention", "symattention", "--pe", "spe",
                                  "--stn", "--direction", "l2r"]),
    ("vanilla_rpe", ["--attention", "vanilla", "--pe", "rpe"]),
    ("symattention_rpe", ["--attention", "symattention", "--pe", "rpe"]),
    ("vanilla_spe_nostn_r2l", ["--attention", "vanilla", "--pe", "spe",
                               "--no-stn", "--direction", "r2l"]),
    ("vanilla_spe_stn_r2l", ["--attention", "vanilla", "--pe", "spe",
                             "--stn", "--direction", "r2l"]),
    ("vanilla_spe_nostn_l2r", ["--attention", "vanilla", "--pe", "spe",
                               "--no-stn", "--direction", "l2r"]),
    ("vanilla_spe_stn_l2r", ["--attention", "vanilla", "--pe", "spe",
                             "--stn", "--direction", "l2r"]),
    ("symattention_spe_nostn_l2r", ["--attention", "symattention", "--pe", "spe",
                                    "--no-stn", "--direction", "l2r"]),
]

# Human-readable Table 8 columns for each cell name.
DESCRIPTIONS = {
    "none_none": ("-", "-", "-"),
    "vanilla_ape": ("Vanilla", "APE", "-"),
    "vanilla_rpe": ("Vanilla", "RPE", "-"),
    "vanilla_spe_nostn_r2l": ("Vanilla", "SPE w/o STN", "R->L"),
    "vanilla_spe_nostn_l2r": ("Vanilla", "SPE w/o STN", "L->R"),
    "vanilla_spe_stn_r2l": ("Vanilla", "SPE", "R->L"),
    "vanilla_spe_stn_l2r": ("Vanilla", "SPE", "L->R"),
    "symattention_ape": ("SymAttention", "APE", "-"),
    "symattention_rpe": ("SymAttention", "RPE", "-"),
    "symattention_spe_nostn_r2l": ("SymAttention", "SPE w/o STN", "R->L"),
    "symattention_spe_nostn_l2r": ("SymAttention", "SPE w/o STN", "L->R"),
    "symattention_spe_stn_r2l": ("SymAttention", "SPE", "R->L"),
    "symattention_spe_stn_l2r": ("SymAttention", "SPE", "L->R"),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one training run per cell per seed. 3 seeds is the minimum that lets you "
                         "say anything about a ~2 AP50 difference")
    ap.add_argument("--cells", nargs="+", default=None,
                    help=f"subset of cell names to run (default: all {len(CELLS)})")
    ap.add_argument("--stack", default="torchvision", choices=["torchvision", "mmdet"])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--project", default="runs/ablation")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke: N batches per epoch")
    ap.add_argument("--delete-weights", action="store_true",
                    help="remove checkpoints after each cell is scored. The AP numbers live in "
                         "results.json; 39 runs x ~300 MB is 12 GB of disk otherwise")
    ap.add_argument("--out", default="results_ablation.md", help="aggregated markdown table")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap.parse_args(argv)


def cell_result_path(project: Path, name: str, seed: int) -> Path:
    return project / f"{name}_s{seed}" / "ablation_result.json"


def run_cell(args, name: str, flags: list[str], seed: int) -> dict | None:
    """Train + score one (cell, seed). Returns its metrics, or None if training failed."""
    project = Path(args.project)
    run_name = f"{name}_s{seed}"
    result_path = cell_result_path(project, name, seed)

    if result_path.is_file():
        payload = json.loads(result_path.read_text())
        print(f"  [skip] {run_name}: already done "
              f"(AP50 {payload.get('AP50', float('nan')):.1f})")
        return payload

    run_dir = project / run_name
    cmd = [
        sys.executable, str(ROOT / "tools" / "train.py"),
        "--data-root", args.data_root,
        "--stack", args.stack,
        "--project", args.project, "--name", run_name, "--exist-ok",
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--seed", str(seed),
        "--num-workers", str(args.num_workers),
    ] + flags
    if args.limit_batches:
        cmd += ["--limit-batches", str(args.limit_batches)]

    print(f"\n{'=' * 78}\nTRAIN {run_name}\n{'=' * 78}")
    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  !! {run_name} failed (exit {proc.returncode}) -- re-run this command to retry")
        return None

    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        print(f"  !! {run_name}: no checkpoint produced")
        return None

    # Score the FINAL epoch, not best.pt: best.pt is selected on val, so its AP is optimistically
    # biased and cannot be compared across cells. This matches report.md's stated convention.
    print(f"\nEVAL {run_name}")
    ev = subprocess.run([
        sys.executable, str(ROOT / "tools" / "val.py"),
        "--weights", str(last), "--data-root", args.data_root,
        "--tag", run_name, "--num-workers", str(args.num_workers), "--no-plots",
    ])
    if ev.returncode != 0:
        print(f"  !! {run_name}: evaluation failed")
        return None

    record = {}
    log = run_dir / "eval_log.jsonl"
    if log.is_file():
        record = json.loads(log.read_text().strip().splitlines()[-1])
    record.update({"cell": name, "seed": seed, "minutes": round((time.time() - t0) / 60, 1)})
    result_path.write_text(json.dumps(record, indent=2))

    if args.delete_weights:
        for p in (run_dir / "weights").glob("*.pt"):
            p.unlink()
        print(f"  removed checkpoints for {run_name}")
    return record


def aggregate(args, names: list[str]) -> str:
    """Collect every finished cell into a mean +/- std markdown table."""
    project = Path(args.project)
    lines = [
        "# Table 8 ablation (local)",
        "",
        f"Category-agnostic TB detection on TB-only val. Stack: `{args.stack}`. "
        f"{args.epochs} epochs, batch {args.batch_size}, lr {args.lr}.",
        f"Seeds: {', '.join(str(s) for s in args.seeds)}. "
        f"Values are the **final-epoch** score (best.pt is val-selected and biased), "
        f"reported as mean +/- std across seeds.",
        "",
        "| Attention | Positional Encoding | Symmetry | AP50 | AP | n |",
        "|---|---|---|---:|---:|---:|",
    ]

    def fmt(values: list[float]) -> str:
        if not values:
            return "-"
        if len(values) == 1:
            return f"{values[0]:.1f}"
        return f"{statistics.mean(values):.1f} +/- {statistics.stdev(values):.1f}"

    any_rows = False
    for name in names:
        ap50s, aps = [], []
        for seed in args.seeds:
            p = cell_result_path(project, name, seed)
            if p.is_file():
                rec = json.loads(p.read_text())
                if rec.get("AP50") is not None:
                    ap50s.append(float(rec["AP50"]))
                    aps.append(float(rec["AP"]))
        if not ap50s:
            continue
        any_rows = True
        att, pe, sym = DESCRIPTIONS.get(name, (name, "", ""))
        lines.append(f"| {att} | {pe} | {sym} | {fmt(ap50s)} | {fmt(aps)} | {len(ap50s)} |")

    if not any_rows:
        lines.append("| *(no completed cells yet)* | | | | | |")

    lines += [
        "",
        "## Reading this",
        "",
        "The paper's claimed ordering is: SymAttention > vanilla at every PE setting; SPE > APE >",
        "RPE; the STN adds a further gain; R->L slightly beats L->R. A difference is only real if it",
        "exceeds the seed-to-seed std in the same row -- with one seed there is no std and no claim",
        "can be made either way.",
    ]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)

    names = args.cells or [n for n, _ in CELLS]
    known = dict(CELLS)
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"ERROR: unknown cell(s): {unknown}")
        print(f"Available: {', '.join(known)}")
        return 2

    total = len(names) * len(args.seeds)
    print(f"Ablation plan: {len(names)} cells x {len(args.seeds)} seed(s) = {total} runs")
    print(f"  stack {args.stack}, {args.epochs} epochs, batch {args.batch_size}")
    print(f"  estimated {total * 12 / 60:.1f}-{total * 18 / 60:.1f} hours on an 8 GB laptop GPU")
    if args.dry_run:
        for name in names:
            print(f"    {name:32s} {' '.join(known[name])}")
        return 0

    t0 = time.time()
    done, failed = 0, []
    for seed in args.seeds:
        for name in names:
            rec = run_cell(args, name, known[name], seed)
            if rec is None:
                failed.append(f"{name}_s{seed}")
            else:
                done += 1
            # rewrite the table after every cell, so a killed sweep still leaves current results
            Path(args.out).write_text(aggregate(args, names), encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"Ablation finished: {done}/{total} runs in {(time.time() - t0) / 3600:.2f} h")
    if failed:
        print(f"  failed: {', '.join(failed)}  (re-run the same command to retry just these)")
    print(f"  table -> {args.out}")
    print(f"{'=' * 78}\n")
    print(Path(args.out).read_text(encoding="utf-8"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
