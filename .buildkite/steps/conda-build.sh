#!/bin/bash

set -xeo pipefail

echo "+++ :snake: :construction_worker: conda build"
conda build . --no-anaconda-upload || exitCode=$?
# if successful, then upload to buildkite
if [ "${exitCode}" == "0" ]; then
  # get the name of the resulting package
  conda_package="$(conda build . --output)"
  # upload to buildkite
  buildkite-agent artifact upload "${conda_package}"
fi

exit $exitCode
