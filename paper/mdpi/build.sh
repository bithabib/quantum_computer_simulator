#!/usr/bin/env bash
# Build the MDPI paper PDF. The MDPI class + logos + .bst live in Definitions/.
# EPS logos are pre-converted to PDF (below) because epstopdf is not installed;
# this makes the build work with a plain pdflatex (no -shell-escape / epstopdf).
set -e
cd "$(dirname "$0")"

# Pre-convert MDPI EPS logos to the -eps-converted-to.pdf names pdflatex expects.
for f in logo-mdpi logo-updates; do
  if [ ! -f "Definitions/${f}-eps-converted-to.pdf" ]; then
    gs -q -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=pdfwrite \
       -sOutputFile="Definitions/${f}-eps-converted-to.pdf" "Definitions/${f}.eps"
  fi
done

pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
echo "Built main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
