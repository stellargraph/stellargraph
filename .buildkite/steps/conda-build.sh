#!/bin/bash

set -xeo pipefail

echo "+++ :snake: Conda build"
conda build . --no-anaconda-upload

