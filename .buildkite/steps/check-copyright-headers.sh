#!/bin/bash
# shellcheck disable=SC2016
# Disable the above because this formats some markdown, and so there's lots of literal ` in strings

set -euo pipefail

temp="$(mktemp)"

copyrightRegex="# Copyright [0-9-]*2020 Data61, CSIRO"

annotate_error() {
  context="$1"
  msg="$2"


  exitCode=1
}

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
  msg="Found files without a copyright header (no matches for \`$copyrightRegex\`)"
  echo "$msg"

  buildkite-agent annotate --context "copyright-existence" --style error << EOF
${msg}:

$(sed 's/\(.*\)/- `\1`/' "$temp")
EOF

  exit 1
else
  echo "all files have a copyright header (have a match for \`$copyrightRegex\`)"
fi
