#!/bin/bash

set -xeo pipefail

echo "+++ :snake: conda build"
conda build . --no-anaconda-upload

