#!/bin/bash

set -euo pipefail

error_file=/tmp/sphinx-errors.txt

cd docs

echo "--- install documentation requirements"
pip install -r requirements.txt

echo "--- listing dependency versions"
pip freeze

echo "+++ building docs"
exit_code=0
make html SPHINXOPTS="-W --keep-going -w $error_file" || exit_code="$?"

if [ "$exit_code" -ne 0 ]; then
  echo "--- annotating build with failures"

  buildkite-agent annotate --context "sphinx-doc-build" --style error <<EOF
The sphinx build had warning(s) and/or error(s) ([failing step](#$BUILDKITE_JOB_ID)):

~~~terminal
$(cat "$error_file")
~~~

These may mean that the documentation doesn't display as expected and so should be fixed.
EOF
  exit 1
fi
