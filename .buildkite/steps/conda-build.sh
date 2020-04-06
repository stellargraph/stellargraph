#!/bin/bash

set -xeo pipefail

echo "+++ :snake: :construction_worker: conda build"
conda build . --no-anaconda-upload

echo "+++ :snake::buildkite: upload package"
conda_package="$(conda build . --output)"
buildkite-agent artifact upload "${conda_package}"
