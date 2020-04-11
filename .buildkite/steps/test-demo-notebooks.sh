#!/bin/bash

set -xeo pipefail

stellargraph_dir="$PWD"

SPLIT="${BUILDKITE_PARALLEL_JOB_COUNT:-1}"
INDEX="${BUILDKITE_PARALLEL_JOB:-0}"

echo "--- :books: collecting notebooks"
# Create array with all notebooks in demo directories, with explicit sorting so every parallel step
# sees the same ordering
cd "${stellargraph_dir}"
IFS=$'\n' read -rd '' -a NOTEBOOKS < <(
  find "${stellargraph_dir}/demos" -name "*.ipynb" | sort
) || true # (the read always exits with 1, because it hits EOF)

NUM_NOTEBOOKS_TOTAL=${#NOTEBOOKS[@]}

if [ "$NUM_NOTEBOOKS_TOTAL" -ne "$SPLIT" ]; then
  msg="Buildkite step parallelism is set to $SPLIT but found $NUM_NOTEBOOKS_TOTAL notebooks. Please update the notebook testing \`parallelism\` setting in \`.buildkite/pipeline.yml\`  to match the number of notebooks!

If a pull request adding or removing notebooks was merged recently, this may be fixed by doing a merge with \`develop\`; for example: \`git fetch origin && git merge origin/develop\`."
  echo "$msg"
  # Every step will do this annotation, but they all use the same context and there's no --append
  # flag here, and so they replace each other. Thus, there should only ever be a single instance of
  # it.
  buildkite-agent annotate --context "incorrect-notebook-parallelism" --style error "$msg"
  exit 1
fi

f=${NOTEBOOKS[$INDEX]}

case $(basename "$f") in
  'attacks_clustering_analysis.ipynb' | 'hateful-twitters-interpretability.ipynb' | 'hateful-twitters.ipynb' | 'stellargraph-attri2vec-DBLP.ipynb' | \
    'node-link-importance-demo-gat.ipynb' | 'node-link-importance-demo-gcn.ipynb' | 'node-link-importance-demo-gcn-sparse.ipynb' | 'rgcn-aifb-node-classification-example.ipynb' | \
    'stellargraph-metapath2vec.ipynb')
    # These notebooks do not yet work on CI:
    # FIXME #818: datasets can't be downloaded
    # FIXME #819: out-of-memory
    # FIXME #849: CI does not have neo4j
    # FIXME #907: socialcomputing.asu.edu is down
    echo "+++ :python: :skull_and_crossbones: skipping $f"
    exit 2 # this will be a soft-fail for buildkite
    ;;

  'loading-saving-neo4j.ipynb' | 'directed-graphsage-on-cora-neo4j-example.ipynb' | 'undirected-graphsage-on-cora-neo4j-example.ipynb' | 'load-cora-into-neo4j.ipynb')
    # these are tested separately (see test-neo4j-notebooks.sh)
    echo "+++ :python: skipping Neo4j notebook $f"
    exit 0
    ;;
esac

echo "--- :python: installing stellargraph"
# install stellargraph itself, which (hopefully) won't install any dependencies
pip install .

echo "--- listing dependency versions"
pip freeze

.buildkite/steps/test-single-notebook.sh "$f"
