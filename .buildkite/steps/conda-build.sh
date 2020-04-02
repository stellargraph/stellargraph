#!/bin/bash

set -xeo pipefail

echo "+++ :snake: :construction_worker: Conda build"
conda build . --no-anaconda-upload
echo "--- "
conda_package="$(conda build . --output)"
# upload to buildkite
buildkite-agent artifact upload "${conda_package}"