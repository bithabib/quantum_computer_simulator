"""Cross-check the response letter's hard-coded references against the
manuscript's actual numbering.

Prints the manuscript's section / table / figure map (from main.pdf via
pdftotext) and every "Sec.~X", "Table~N", "Fig.~N" mention in the response
letter and cover letter, with the caption the number currently points to, so
mismatches are easy to spot after floats move.

    python paper/qmi/check_refs.py
"""

import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def pdf_text(name):
    return subprocess.run(["pdftotext", "-layout", os.path.join(HERE, name), "-"],
                          capture_output=True, text=True).stdout


def main():
    t = pdf_text("main.pdf")
    tables = dict(re.findall(r"^\s*Table (\d+) ([A-Z][^\n]{0,70})", t, re.M))
    figs = dict(re.findall(r"^\s*Fig\. (\d+) ([A-Z][^\n]{0,70})", t, re.M))
    secs = dict(re.findall(r"^(\d+(?:\.\d+)?) ([A-Z][^\n]{0,60})$", t, re.M))
    print("== manuscript numbering ==")
    for k in sorted(secs, key=lambda x: [int(p) for p in x.split(".")]):
        print("  Sec %-5s %s" % (k, secs[k]))
    for k in sorted(tables, key=int):
        print("  Table %-3s %s" % (k, tables[k]))
    for k in sorted(figs, key=int):
        print("  Fig %-3s %s" % (k, figs[k]))
    for name in ["response_to_reviewers.tex", "cover_letter.tex"]:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        # drop quoted reviewer text (\rev{...}) so their original numbering is ignored
        src = re.sub(r"\\rev\{.*?\}\n", "", src, flags=re.S)
        print("\n== references in %s (outside reviewer quotes) ==" % name)
        for m in re.finditer(r"(Sec|Table|Tables|Fig|Figs)\.?~?\s*(S?\d+(?:\.\d+)?)", src):
            kind, num = m.group(1), m.group(2)
            if num.startswith("S"):
                target = "(supplement)"
            elif kind.startswith("Sec"):
                target = secs.get(num, "?? no such section")
            elif kind.startswith("Table"):
                target = tables.get(num, "?? no such table")
            else:
                target = figs.get(num, "?? no such figure")
            line = src[:m.start()].count("\n") + 1
            print("  line %4d  %-6s %-5s -> %s" % (line, kind, num, target[:60]))


if __name__ == "__main__":
    main()
