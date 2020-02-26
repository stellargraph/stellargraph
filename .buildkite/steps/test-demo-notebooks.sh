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
    'directed-graphsage-on-cora-neo4j-example.ipynb' | 'undirected-graphsage-on-cora-neo4j-example.ipynb' | 'load-cora-into-neo4j.ipynb' | \
    'stellargraph-metapath2vec.ipynb')
    # These notebooks do not yet work on CI:
    # FIXME #818: datasets can't be downloaded
    # FIXME #819: out-of-memory
    # FIXME #849: CI does not have neo4j
    # FIXME #907: socialcomputing.asu.edu is down
    echo "+++ :python: :skull_and_crossbones: skipping $f"
    exit 2 # this will be a soft-fail for buildkite
    ;;
esac

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

echo "+++ :jupyter: making result viewable"
filename="$(basename "$f")"
# and also upload the notebook with outputs, for someone to view by downloading or via nbviewer; we
# can include a link to the latter automatically
buildkite-agent artifact upload "$filename" 2>&1 | tee agent-output.txt

# extract the artifact UUID: the output is '.... Uploading artifact <UUID> <filename> (<size> ...',
# so match the '<UUID> <filename>' section (to be sure it's the correct ID), and then cut out the
# UUID part. The UUID is formatted as a conventional hex UUID 123e4567-e89b-12d3-a456-426655440000,
# and is matched with a relaxed regex (any hex digits and -). The filename needs to be a literal
# match, and so needs to have any special characters escaped.
re_safe_filename="$(printf '%s' "$filename" | sed 's/[.[\*^$]/\\&/g')"
artifact_id="$(grep --only-matching "[0-9a-f-]* $re_safe_filename" agent-output.txt | cut -f1 -d' ')"

if [ -z "$artifact_id" ]; then
  echo "failed to find artifact ID; this may be an error in the script ($0)"
  exit 1
fi

url="https://nbviewer.jupyter.org/urls/buildkite.com/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds/${BUILDKITE_BUILD_NUMBER}/jobs/${BUILDKITE_JOB_ID}/artifacts/${artifact_id}"
echo "This notebook can be viewed at <$url>"

if [ "$exitCode" -ne 0 ]; then
  # the notebook failed, so let's flag that more obviously, with helpful links
  buildkite-agent annotate --style "error" --context "$filename" << EOF
Notebook \`$filename\` had an error: [failed job](#${BUILDKITE_JOB_ID}), [rendered notebook]($url)
EOF
fi

exit $exitCode
