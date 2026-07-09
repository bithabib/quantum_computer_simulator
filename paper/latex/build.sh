#!/usr/bin/env bash
# Build the paper PDF. IEEEtran.cls and IEEEtran.bst are bundled in this dir so
# no system texlive-publishers install is required.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
echo "Built main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
