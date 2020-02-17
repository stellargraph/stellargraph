#!/bin/bash

set -xeo pipefail

stellargraph_dir="$PWD"

SPLIT="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"
INDEX="${BUILDKITE_PARALLEL_JOB:-0}"

echo "--- :books: collecting notebooks"
# Create array with all notebooks in demo directories.
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

f=${NOTEBOOKS[$INDEX]}

case $(basename "$f") in
  'attacks_clustering_analysis.ipynb' | 'hateful-twitters-interpretability.ipynb' | 'hateful-twitters.ipynb' | 'stellargraph-attri2vec-DBLP.ipynb' | \
    'node-link-importance-demo-gat.ipynb' | 'node-link-importance-demo-gcn.ipynb' | 'node-link-importance-demo-gcn-sparse.ipynb' | 'rgcn-aifb-node-classification-example.ipynb')
    # These notebooks do not yet work on CI:
    # FIXME #818: datasets can't be downloaded
    # FIXME #819: out-of-memory
    echo "+++ :python: :skull_and_crossbones: skipping $f"
    exit 2 # this will be a soft-fail for buildkite
    ;;
esac

echo "--- :python: installing papermill"
# Pulling in https://github.com/nteract/papermill/pull/459 for --execution-timeout, which hasn't been released yet
pip install https://github.com/nteract/papermill/archive/master.tar.gz

echo "--- :python: installing stellargraph"
# install stellargraph itself, which (hopefully) won't install any dependencies
pip install .

echo "--- listing dependency versions"
pip freeze

echo "+++ :python: running $f"
cd "$(dirname "$f")"
# run the notebook, saving it back to where it was, printing everything
exitCode=0
# papermill will replace parameters on some notebooks to make them run faster in CI
papermill --execution-timeout=600 --parameters_file "${stellargraph_dir}/.buildkite/notebook-parameters.yml" --log-output "$f" "$f" || exitCode=$?

# and also upload the notebook with outputs, for someone to download and look at
buildkite-agent artifact upload "$(basename "$f")"
exit $exitCode
