#!/usr/bin/env bash
# Regenerate the SVG images in docs/diagrams/ from the Mermaid blocks in docs/architecture.md.
# The Mermaid source is the single source of truth: edit it there, then run this script.
#
#   bash docs/diagrams/render.sh
#
# Needs Node (npx). If Puppeteer cannot find Chrome, point it at an installed one, e.g.
#   echo '{"executablePath": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}' > /tmp/pp.json
#   MERMAID_PUPPETEER_CONFIG=/tmp/pp.json bash docs/diagrams/render.sh
set -euo pipefail

DIAGRAMS_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$DIAGRAMS_DIR/../architecture.md"
MMDC_VERSION="11.4.2"

# One name per Mermaid block, in order; "-" skips a block (the colour key).
NAMES=(- 01-big-picture 02-analyze 03-forecast-agents 04-review 05-learning-loop 06-backtest
       07-safety-guards 08-storage)

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Plain SVG text labels instead of embedded HTML, so any image viewer shows the text
echo '{"htmlLabels": false, "flowchart": {"htmlLabels": false}}' > "$WORK_DIR/config.json"

awk -v dir="$WORK_DIR" '
    /^```mermaid/ { n++; out = sprintf("%s/block-%02d.mmd", dir, n); inside = 1; next }
    /^```/ && inside { inside = 0; close(out); next }
    inside { print > out }
' "$SOURCE"

blocks=("$WORK_DIR"/block-*.mmd)
if [ "${#blocks[@]}" -ne "${#NAMES[@]}" ]; then
    echo "Expected ${#NAMES[@]} Mermaid blocks in $SOURCE, found ${#blocks[@]}; update NAMES." >&2
    exit 1
fi

PUPPETEER_ARGS=()
if [ -n "${MERMAID_PUPPETEER_CONFIG:-}" ]; then
    PUPPETEER_ARGS=(-p "$MERMAID_PUPPETEER_CONFIG")
fi

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    [ "$name" = "-" ] && continue
    npx -y "@mermaid-js/mermaid-cli@$MMDC_VERSION" "${PUPPETEER_ARGS[@]+"${PUPPETEER_ARGS[@]}"}" \
        -q -b white -c "$WORK_DIR/config.json" -i "${blocks[$i]}" -o "$DIAGRAMS_DIR/$name.svg"
    echo "wrote docs/diagrams/$name.svg"
done
