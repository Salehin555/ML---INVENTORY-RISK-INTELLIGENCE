"""Compile the workflow diagram (responses/FIG_WORKFLOW.tex) as a standalone PDF: python wf_preview.py OUTDIR"""
import os, sys, re, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "figures")
wf = open(os.path.join(R, "responses", "FIG_WORKFLOW.tex"), encoding="utf-8").read()
tikz = wf[wf.index("\\providecommand{\\wfnohyph}"): wf.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")]
doc = ("\\documentclass[border=4pt]{standalone}\n\\usepackage{tikz}\n\\usepackage{amsmath,amssymb}\n"
       "\\usetikzlibrary{positioning,arrows.meta}\n\\begin{document}\n" + tikz + "\n\\end{document}\n")
open(os.path.join(out, "fig_workflow.tex"), "w", encoding="utf-8").write(doc)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig_workflow.tex"], cwd=out, capture_output=True)
for ext in (".aux", ".log", ".tex"):
    p = os.path.join(out, "fig_workflow" + ext)
    if os.path.exists(p): os.remove(p)
print("ok" if os.path.exists(os.path.join(out, "fig_workflow.pdf")) else "failed")
