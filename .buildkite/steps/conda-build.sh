#!/bin/bash

set -xeo pipefail

echo "+++ :snake: :construction_worker: Conda build"
conda build . --no-anaconda-upload

