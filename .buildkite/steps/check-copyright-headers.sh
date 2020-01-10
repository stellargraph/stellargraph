#!/bin/bash

set -euo pipefail

exitCode=0
temp="$(mktemp)"

copyrightRegex="# Copyright [0-9-]*2020 Data61, CSIRO"

echo "--- checking files have copyright headers"
# Some files shouldn't have our copyright header and so are ignored
find . \( \
  -name "*.py" \
  -a ! -path "./demos/link-prediction/random-walks/utils/node2vec/node2vec.py" \
  -a ! -path "./demos/link-prediction/random-walks/utils/node2vec/main.py" \
  -a ! -path "./docs/conf.py" \
  \) \
  -exec grep -L "$copyrightRegex" {} + | tee "$temp"

if [ -s "$temp" ]; then
  echo "^^^ +++"
  echo "found files without a copyright header (no matches for /$copyrightRegex/)"
  exit 1
else
  echo "all files have a copyright header (have a match for /$copyrightRegex/)"
fi
