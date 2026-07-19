"""Fill the result macros in main.tex from qml_bp.train output.

Runs the trainer on a dataset, parses its stdout, and rewrites the
`\newcommand{...}{TBD}`-style result macros in main.tex so the paper prose and
tables stay in sync with the actual model numbers.

    python paper/mdpi/fill_results.py --data data_bp/bp_dataset.csv

Idempotent: re-run any time the dataset or model changes.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# The manuscript defines its result macros once, so we
# patch whichever of them exist.
MAIN_TEX_FILES = [
    os.path.join(HERE, "main.tex"),
]


def run_trainer(data):
    cmd = [sys.executable, "-m", "qml_bp.train", "--data", data]
    print("Running:", " ".join(cmd))
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    print(out.stdout)
    return out.stdout


def parse(out):
    """Extract macro name -> value strings from trainer stdout."""
    v = {}

    def grab(pat, key, cast=lambda s: s):
        m = re.search(pat, out)
        if m:
            v[key] = cast(m.group(1))

    grab(r"(\d+(?:\.\d+)?)% global-cost", "GlobalFrac")
    grab(r"barren fraction .*?: (\d+(?:\.\d+)?)%", "BarrenFrac")

    # Random hold-out block
    hold = re.search(r"=== Random hold-out split ===(.*?)===", out, re.S)
    if hold:
        b = hold.group(1)
        m = re.search(r"R2 = (-?\d+\.\d+)\s+MAE = (\d+\.\d+)", b)
        if m:
            v["RsqHold"], v["MaeHold"] = m.group(1), m.group(2)
        m = re.search(r"accuracy = (\d+\.\d+)(?:\s+AUC = (\d+\.\d+))?", b)
        if m:
            v["AccHold"] = m.group(1)
            if m.group(2):
                v["AucHold"] = m.group(2)

    # Extrapolation block
    ex = re.search(
        r"=== Extrapolation split \(train n<=(\d+).*?===(.*?)(?:===|$)", out, re.S
    )
    if ex:
        v["ExtrapCut"] = ex.group(1)
        b = ex.group(2)
        m = re.search(r"R2 = (-?\d+\.\d+)\s+MAE = (\d+\.\d+)", b)
        if m:
            v["RsqExtrap"], v["MaeExtrap"] = m.group(1), m.group(2)
        m = re.search(r"train rows (\d+), test rows (\d+)", b)
        if m:
            # Use a LaTeX thin space (\,) as the thousands separator -- crucially
            # brace-free, so the macro value contains no '}' that would confuse
            # the patch regex on a re-run.
            v["ExtrapTrainRows"] = "{:,}".format(int(m.group(1))).replace(",", "\\,")
            v["ExtrapTestRows"] = "{:,}".format(int(m.group(2))).replace(",", "\\,")

    return v


def patch(values):
    for main_tex in MAIN_TEX_FILES:
        if not os.path.exists(main_tex):
            continue
        with open(main_tex) as fh:
            tex = fh.read()
        n = 0
        for key, val in values.items():
            # Match the macro's value group, tolerating one level of nested
            # braces (e.g. a value like 20{,}000) so a re-run cannot corrupt it.
            pat = (r"(\\newcommand\{\\" + key
                   + r"\}\{)(?:[^{}]|\{[^{}]*\})*(\})")
            tex, k = re.subn(pat, lambda m: m.group(1) + val + m.group(2), tex)
            n += k
        with open(main_tex, "w") as fh:
            fh.write(tex)
        print("Patched %d macros in %s" % (n, main_tex))
    for key, val in values.items():
        print("  \\%s = %s" % (key, val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    out = run_trainer(args.data)
    values = parse(out)
    if not values:
        sys.exit("Could not parse any results from trainer output.")
    patch(values)


if __name__ == "__main__":
    main()
