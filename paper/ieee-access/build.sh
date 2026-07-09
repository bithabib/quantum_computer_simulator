#!/usr/bin/env bash
# Build the IEEE Access paper. Note: the (old) ieeeaccess.cls emits a harmless
# "Missing number, treated as zero" during \maketitle (title-page spacing
# rounding), which makes pdflatex exit non-zero even though the PDF is fine --
# so we do NOT use `set -e` and instead check that main.pdf was produced.
cd "$(dirname "$0")"

# 1) Render the TikZ pipeline diagram as a standalone PDF (ieeeaccess.cls is
#    incompatible with modern PGF, so we don't load TikZ in the main document).
if [ ! -f figs/pipeline.pdf ] || [ pipeline_src.tex -nt figs/pipeline.pdf ] \
   || [ fig_pipeline.tex -nt figs/pipeline.pdf ]; then
  pdflatex -interaction=nonstopmode pipeline_src.tex > /dev/null 2>&1 || true
  [ -f pipeline_src.pdf ] && cp pipeline_src.pdf figs/pipeline.pdf
fi

# 2) Main document (pdflatex -> bibtex -> pdflatex x2).
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > main.build.log 2>&1 || true

if [ -f main.pdf ]; then
  echo "Built main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
  u=$(grep -ic "Reference.*undefined\|Citation.*undefined" main.build.log)
  echo "  undefined references/citations: $u"
else
  echo "BUILD FAILED -- see main.build.log"; exit 1
fi
