#!/bin/bash

set -xeo pipefail

echo "--- conda build"
pip freeze

conda build .
