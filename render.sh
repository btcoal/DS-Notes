# render cheatsheets

# command line flags

INPUT="$1"
OUTPUT="$2"

pandoc -s performance-metrics.md -f gfm+tex_math_dollars --pdf-engine=xelatex -o performance-metrics.pdf
pandoc -s performance-metrics.md -f gfm+tex_math_dollars -t html5 --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js -o performance-metrics.html


pandoc -s "$INPUT" \
  -f gfm+tex_math_dollars \
  --pdf-engine=xelatex \
  -o "$OUTPUT"