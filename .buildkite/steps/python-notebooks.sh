#!/bin/bash

set -xeo pipefail

SPLIT="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"
INDEX="${BUILDKITE_PARALLEL_JOB:-0}"

echo "--- :books: collecting notebooks"
# Create array with all notebooks
NOTEBOOKS=()
while IFS= read -r -d $'\0'; do
  NOTEBOOKS+=("$REPLY")
done < <(find "$PWD" -name "*.ipynb" -print0)

NUM_NOTEBOOKS_TOTAL=${#NOTEBOOKS[@]}

echo "--- :python: installing notebooks dependencies"
pip install -q --no-cache-dir -r requirements.txt -e .

echo "--- :python: listing notebooks dependencies"
pip freeze

echo "--- :python: preparing to run notebooks"

exitCode=0
for f in "${NOTEBOOKS[@]}"; do
  echo "--- :python: running $f"
  cd "$(dirname "$f")"
  # run the notebook, saving it back to where it was, printing everything
  papermill --log-output "$f" "$f" || {
    exitCode=$?
    # this section failed, so open it by default
    echo "^^^ +++"
  }

  # and also upload the notebook with outputs, for someone to download and look at
  buildkite-agent artifact upload "$(basename "$f")"
done
# any failed?
exit $exitCode
