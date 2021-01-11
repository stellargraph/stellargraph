#!/bin/bash
# shellcheck disable=SC2016
# Disable the above because this formats some markdown, and so there's lots of literal ` in strings

set -xeuo pipefail

copyrightRegex="# Copyright [0-9-]*2020 Data61, CSIRO"

echo "--- checking files have copyright headers"
# Some files shouldn't have our copyright header and so are ignored
incorrect=$(find . \( \
  -name "*.py" \
  -a ! -path "./docs/conf.py" \
  \) \
  -exec grep -L "$copyrightRegex" {} +)

echo "$incorrect"

if [ -n "$incorrect" ]; then
  echo "^^^ +++"
  msg="Found files without a copyright header (no matches for \`$copyrightRegex\`)"
  echo "$msg"
  exit 1
else
  echo "all files have a copyright header (have a match for \`$copyrightRegex\`)"
fi
