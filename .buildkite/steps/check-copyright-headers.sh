#!/bin/bash

set -euo pipefail

exitCode=0
temp="$(mktemp)"

copyrightRegex="# Copyright [0-9-]* Data61, CSIRO"

annotate_error() {
  context="$1"
  msg="$2"

  echo "^^^ +++"
  echo "$msg"

  buildkite-agent annotate --context "$context" --style error << EOF
${msg}:

$(sed 's/^\(.*\)$/- `\1`/' "$temp")
EOF

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
  annotate_error "copyright-existence" "Found files without a copyright header (no matches for /$copyrightRegex/)"
else
  echo "all files have a copyright header (have a match for /$copyrightRegex/)"
fi

echo "--- checking copyright headers are up to date"
# look for files where their copyright header (grep $copyrightRegex) doesn't mention the current
# year (grep -v $year), and then check (while read ....) whether there's been any commit touching
# that file in the current year.
year="$(date +%Y)"

find . -name "*.py" -exec grep "$copyrightRegex" {} + | grep -v "$year" | while read -r fileAndLine; do
  # grep prints `<filename>:<matching line>`, so remove everything from the first : to get the
  # filename
  filename="${fileAndLine%%:*}"
  # print the year (%Y) of the author dates (%ad) of each commit to the file in question, and then
  # find the largest (most recent) one
  lastModifiedYear=$(git log --pretty=tformat:%ad --date=format:%Y -- "$filename" | sort -nr | head -1)
  if [ "$lastModifiedYear" -eq "$year" ]; then
    # the file has been modified this year, but the copyright header doesn't mention it, uh oh!
    # Print the line for follow up
    echo "$fileAndLine"
  fi
done | tee "$temp"

if [ -s "$temp" ]; then
  annotate_error "copyright-year" "Found files modified in $year (according to git) without $year in their copyright header"
else
  echo "all files modified in $year (according to git) have $year in their copyright header"
fi

exit "$exitCode"
