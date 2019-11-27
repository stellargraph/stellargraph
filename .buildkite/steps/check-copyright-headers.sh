#!/bin/sh

set -eu

exitCode=0
temp="$(mktemp)"

copyrightRegex="# Copyright [0-9-]* Data61, CSIRO"
echo "--- checking files have copyright headers"
find . -name "*.py" -exec grep -L "$copyrightRegex" {} + | tee "$temp"

if [ -s "$temp" ]; then
  echo "^^^ +++"
  echo "found files without copyright header matching $copyrightRegex"
  exitCode=1
fi


echo "--- checking copyright headers are up to date"
# look for files that have been modified in the last year (-mtime -$dayOfYear), where their
# copyright header (grep $copyrightRegex) doesn't mention the current year (grep -v $year).
year="$(date +%Y)"
dayOfYear="$(date +%j)"

find . -name "*.py" -mtime "-$dayOfYear" -exec grep "$copyrightRegex" {} + | grep -v "$year" | tee "$temp"

if [ -s "$temp" ]; then
  echo "^^^ +++"
  echo "found files modified in $year (in the last $dayOfYear days) without $year in their copyright header"
  exitCode=1
fi

exit "$exitCode"
