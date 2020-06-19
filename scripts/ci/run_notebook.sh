#!/bin/bash

set -euo pipefail

notebook="$1"
# papermill will replace parameters on some notebooks to make them run faster in CI
papermill --execution-timeout=600 \
          --parameters_file "./.buildkite/notebook-parameters.yml" --log-output \
          "$notebook" "$notebook"

notebook_name="$(basename "$notebook")"
echo "::set-output name=notebook_name::$notebook_name"
