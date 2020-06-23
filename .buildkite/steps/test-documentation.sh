#!/bin/bash

set -euo pipefail

# need two files for the -w, because it overwrites not appends
error_file=/tmp/sphinx-errors.txt
spelling_error_file=/tmp/sphinx-spelling-errors.txt

spelling_file="_build/spelling/output.txt"
opts="-W --keep-going"

cd docs

echo "--- installing pandoc"
apt update
apt install -y pandoc enchant

echo "--- installing documentation requirements"
pip install -r requirements.txt

echo "--- listing dependency versions"
pip freeze

exit_code=0
echo "+++ building docs"
make html SPHINXOPTS="$opts -w $error_file" || exit_code="$?"

echo "+++ checking spelling"
make spelling SPHINXOPTS="$opts -w $spelling_error_file" || exit_code="$?"

if [ "$exit_code" -ne 0 ]; then
  exit 1
fi
