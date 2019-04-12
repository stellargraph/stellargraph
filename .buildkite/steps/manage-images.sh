#!/bin/bash
set -euo pipefail

build_push() {
  local image="$1"
  local tag="$2"

  echo "Building ${image}";
  docker/"${image}"/build.sh
  docker tag stellargraph/"${image}" stellargraph/"${image}":"${tag}"
  docker push stellargraph/"${image}":"${tag}"
}

build_publish_images() {
  local tag="$1"
  shift

  for i in "$@"; do
    echo "+++ :docker: build and push $i"
    build_push "$i" "$tag"
  done
}

images=(
stellargraph
stellargraph-demos
)

action="$1"

case "${BUILDKITE_BRANCH}" in
  develop)
    version=latest
  feature/*)
    version=testing
esac

case "${action}" in
  build-publish)
    build_publish_images "${version}" "${images[@]}"
    ;;
  *)
    echo "[ERROR] unknown argument '${action}'" >&2
    exit 1
esac
