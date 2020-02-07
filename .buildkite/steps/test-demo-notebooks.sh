#!/bin/bash

set -xeo pipefail

stellargraph_dir="$PWD"

SPLIT="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"
INDEX="${BUILDKITE_PARALLEL_JOB:-0}"

echo "--- :books: collecting notebooks"
# Create array with all notebooks in notebooks' directories. Notebooks are
# available in /home/hadoop/scenarios and /home/hadoop/demos on the edgenode
# with correct permissions for user hadoop.
cd "${stellargraph_dir}"
NOTEBOOKS=()
while IFS= read -r -d $'\0'; do
  NOTEBOOKS+=("$REPLY")
done < <(find "${stellargraph_dir}/demos" -name "*.ipynb" -print0)

NUM_NOTEBOOKS_TOTAL=${#NOTEBOOKS[@]}

if [ "$NUM_NOTEBOOKS_TOTAL" -ne "$SPLIT" ]; then
  echo "Buildkite step parallelism is set to $SPLIT but found $NUM_NOTEBOOKS_TOTAL notebooks. Please update parallelism to match the number of notebooks!"
  exit 1
fi


echo "--- installing dependencies"
pip install -q --no-cache-dir '.[demos,igraph]'

echo "--- listing dependency versions"
pip freeze


f=${NOTEBOOKS[$INDEX]}

echo "+++ :python: running $f"
case $(basename "$f") in
  'Scenario2 (1.b) Distributed - Twitter.ipynb')
    # FIXME: these notebooks do not yet work on CI (#2261)
    ;;
  *) # fine, run this one
    cd "$(dirname "$f")"
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=2400 "$f"
    ;;
esac
