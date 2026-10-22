"""Finds the page of each heading in the PDF and stores it in toc_pages.json (used by build_report.py on the next run)."""
import json, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
pdf = here / "project_report.pdf"
n = int([l for l in subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages")][0].split()[1])
pages = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(pdf), "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
src = (here / "build_report.py").read_text()
entries = json.loads(src[src.index("TOC_ENTRIES = ") + 14: src.index("]", src.index("TOC_ENTRIES = ")) + 1].replace("\n", " ").replace("'", '"')) if False else None
import re
block = src[src.index("TOC_ENTRIES = ["): src.index("]", src.index("TOC_ENTRIES = [")) + 1]
entries = re.findall(r'"([^"]+)"', block)
out = {}
for e in entries:
    for k, txt in enumerate(pages, 1):
        if k <= 3 and e not in ("Title Page", "Abstract", "Table of Contents"):
            continue
        lines = [l.strip() for l in txt.splitlines()]
        if e in lines and not any(e + "\t" in l or l.endswith(" " + str(k)) and False for l in lines):
            if e in ("Title Page",): out[e] = 1; break
            out[e] = k; break
out["Title Page"] = 1
(here / "toc_pages.json").write_text(json.dumps(out, indent=1))
print(out)
