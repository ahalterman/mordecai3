"""Run the e52 arm grid and print one comparison table.

Arms (frozen checkpoint, no retraining -- only the candidate list changes):
  base      untouched pickles, the reference
  abbrev    R1 mention normalisation before the ES query
  demote_h  R2 defunct row superseded by a live twin
  junk      R4 bare unreferenced exact-name row shadowing a notable twin
  dedupe_p  R3 collapse co-located same-name rows, keep the P side
  dedupe_a  R3 collapse, keep the A side
  ship      the rules the census actually supports

Two seeds (101 = ship, 42) x two windows (100 = serving, 500 = ledger).

    python run_grid.py --arms base,abbrev,demote_h --seeds 101,42 --windows 100
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
PY = os.path.join(ROOT, ".venv", "bin", "python")

ARMS = {
    "base": [],
    "abbrev": ["--rules", "abbrev"],
    "demote_h": ["--rules", "demote_h"],
    "junk": ["--rules", "demote_junk"],
    "dedupe_p": ["--rules", "dedupe", "--dedupe-keep", "p"],
    "dedupe_a": ["--rules", "dedupe", "--dedupe-keep", "a"],
    "ship": ["--rules", "abbrev,demote_h,demote_junk"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,abbrev,demote_h,junk,ship")
    ap.add_argument("--seeds", default="101,42")
    ap.add_argument("--windows", default="100,500")
    ap.add_argument("--sources", default="TR,LGL,GWN,WikiDocs,Prodigy,Synth")
    a = ap.parse_args()

    for seed in a.seeds.split(","):
        for w in a.windows.split(","):
            for arm in a.arms.split(","):
                out = os.path.join(HERE, "res_%s_s%s_w%s.json" % (arm, seed, w))
                if os.path.exists(out):
                    print("skip (exists)", os.path.basename(out))
                    continue
                cmd = [PY, os.path.join(HERE, "eval_hygiene.py"),
                       "--checkpoint",
                       os.path.join(ROOT, "experiments/e29_swa_ep15",
                                    "seed%s.pt" % seed),
                       "--window", w, "--sources", a.sources,
                       "--out", out] + ARMS[arm]
                print(">>", arm, "seed", seed, "window", w, flush=True)
                r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-3000:])
                    print(r.stderr[-3000:])
                    sys.exit("arm %s failed" % arm)
                print(r.stdout.strip().splitlines()[-3], flush=True)
                with open(out.replace(".json", ".log"), "w") as f:
                    f.write(r.stdout)


if __name__ == "__main__":
    main()
