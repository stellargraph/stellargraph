#!/bin/bash

set -xeo pipefail
exitCode=0

echo "+++ :snake: :construction_worker: conda build"
conda build . --no-anaconda-upload || exitCode=$?
# if successful, then upload to buildkite
if [ "${exitCode}" == "0" ]; then
  conda_package="$(conda build . --output)"
  buildkite-agent artifact upload "${conda_package}"
fi

exit $exitCode
