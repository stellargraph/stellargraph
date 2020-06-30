#!/bin/bash

set -euo pipefail

notebook="$1"

# work out the name of the notebook for the artifact uploading, before running the notebook itself,
# in case that fails
notebook_name="$(basename "$notebook")"
echo "::set-output name=notebook_name::$notebook_name"

# papermill will replace parameters on some notebooks to make them run faster in CI
papermill --execution-timeout=600 \
  --parameters_file "./.buildkite/notebook-parameters.yml" --log-output \
  "$notebook" "$notebook"
