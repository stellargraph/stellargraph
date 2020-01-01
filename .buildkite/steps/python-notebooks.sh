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

if [ "$NUM_NOTEBOOKS_TOTAL" -ne "$SPLIT" ]; then
  echo "Buildkite step parallelism is set to $SPLIT but found $NUM_NOTEBOOKS_TOTAL notebooks. Please update parallelism to match the number of notebooks!"
  exit 1
fi

echo "--- :python: installing notebooks dependencies"
pip install -q --no-cache-dir -r requirements.txt -e .

echo "--- :python: listing notebooks dependencies"
pip freeze

echo "--- :python: preparing to run notebooks"

f=${NOTEBOOKS[$INDEX]}

echo "+++ :python: running $f"
cd "$(dirname "$f")"
# run the notebook, saving it back to where it was, printing everything
exitCode=0
papermill --log-output "$f" "$f" || exitCode=$?

# and also upload the notebook with outputs, for someone to download and look at
buildkite-agent artifact upload "$(basename "$f")"
exit $exitCode

