#!/bin/bash
# shellcheck disable=SC2016
# Disable the above because this formats some markdown, and so there's lots of literal ` in strings

set -euo pipefail

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
  msg="Found files without a copyright header (no matches for \`$copyrightRegex\`)"
  echo "$msg"

  # create a markdown list containing each of the files, wrapped in backticks so they're formatted
  # nicely
  markdown_file_list="$(sed 's/\(.*\)/- `\1`/' "$temp")"

  buildkite-agent annotate --context "copyright-existence" --style error << EOF
${msg}:

${markdown_file_list}
EOF

  exit 1
else
  echo "all files have a copyright header (have a match for \`$copyrightRegex\`)"
fi
