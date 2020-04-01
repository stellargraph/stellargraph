#!/bin/bash

set -xeo pipefail

echo "--- conda build"
conda build . --no-anaconda-upload

