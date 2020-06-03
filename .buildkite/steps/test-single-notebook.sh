#!/bin/bash

set -xeo pipefail

stellargraph_dir="$PWD"
f="$1"
extra_info="${2-}"

echo "+++ :python: running $f"
cd "$(dirname "$f")"
# run the notebook, saving it back to where it was, printing everything
exitCode=0
# papermill will replace parameters on some notebooks to make them run faster in CI
papermill --execution-timeout=600 --report-mode --parameters_file "${stellargraph_dir}/.buildkite/notebook-parameters.yml" --log-output "$f" "$f" || exitCode=$?

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
  buildkite-agent annotate --style "error" --context "$filename-${BUILDKITE_JOB_ID}" << EOF
Notebook \`$filename\` had an error${extra_info}: [failed job](#${BUILDKITE_JOB_ID}), [rendered notebook]($url)
EOF
fi

exit $exitCode
