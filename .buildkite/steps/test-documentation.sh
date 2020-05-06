#!/bin/bash

set -euo pipefail

error_file=/tmp/sphinx-errors.txt

cd docs

echo "--- installing pandoc"
apt update
apt install -y pandoc

echo "--- installing documentation requirements"
pip install -r requirements.txt

echo "--- listing dependency versions"
pip freeze

echo "+++ building docs"
exit_code=0
make html SPHINXOPTS="-W --keep-going -w $error_file" || exit_code="$?"

if [ "$exit_code" -ne 0 ]; then
  echo "--- annotating build with failures"

  # strip out the /workdir/ references, so that the filenames are more relevant to the user
  # (relative to the repo root)
  output="$(sed s@/workdir/@@ "$error_file")"

  buildkite-agent annotate --context "sphinx-doc-build" --style error << EOF
The sphinx build had warnings and/or errors. These may mean that the documentation doesn't display as expected and so should be fixed.

~~~terminal
$output
~~~

([View all output](#$BUILDKITE_JOB_ID))
EOF
  exit 1
fi
