#!/bin/bash

set -eou pipefail

echo "+++ installing dependencies"

PIP_FLAGS=("--no-cache-dir")
if [ "$PRERELEASE_VERSIONS" = 1 ]; then
  # install release candidates and beta (etc.) to catch potential future breakage
  PIP_FLAGS+=("--pre")
fi

# install stellargraph without any source code to install its dependencies, and then immediately
# uninstall it (without uninstalling the dependencies), so that the installation with source
# code below will work without the `--upgrade` flag. This flag will cause pip to try to update
# dependencies, which we don't want to happen in that second step.
pip install "${PIP_FLAGS[@]}" '/build/[test,demos]'
pip uninstall -y stellargraph
