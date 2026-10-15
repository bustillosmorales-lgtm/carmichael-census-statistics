"""Copy the quoted cells of the tables of the second revision from the manuscript source into
verify_r2_quoted.json, which verify_r2.py compares with the outputs in results/.

Usage: python tools/extract_quoted_r2.py <manuscript.tex>"""
import json
import os
import re
import sys

LABELS = ["tab:level", "tab:rule", "tab:windows", "tab:counts", "tab:tests", "tab:generators", "tab:decades", "tab:strata",
          "tab:external", "tab:external-r", "tab:digits", "tab:digits24", "tab:power"]


def clean(cell):
    cell = re.sub(r"\\rev\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace("{,}", ",").replace("$", "").replace("\\ast", "*").replace("^*", "*")
    cell = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", cell)
    return cell.strip()


def table_rows(tex, label):
    i = tex.index("\\label{" + label + "}")
    a = tex.index("\\midrule", i) + len("\\midrule")
    b = tex.index("\\bottomrule", a)
    rows = []
    for line in tex[a:b].split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if line:
            rows.append([clean(c) for c in line.split("&")])
    return rows


tex = open(sys.argv[1], encoding="utf-8").read()
out = {lab: table_rows(tex, lab) for lab in LABELS}
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verify_r2_quoted.json")
json.dump({"source": os.path.basename(sys.argv[1]), "tables": out}, open(path, "w", encoding="utf-8"), indent=1)
for k, v in out.items():
    print(f"{k:16s} {len(v)} rows")
