#!/bin/bash
set -euo pipefail

build_push() {
  local image="$1"
  local tag="$2"

  echo "Building ${image}"
  docker/"${image}"/build.sh
  docker tag stellargraph/"${image}" stellargraph/"${image}":"${tag}"
  docker push stellargraph/"${image}":"${tag}"
}

build_publish_numbered() {
  local tag="$1"
  shift

  for i in "$@"; do
    echo "+++ :docker: build and push $i"
    build_push "$i" "$tag"
  done
}

publish_images() {
  local numbered_tag="$1"
  local published_tag="$2"
  shift 2

  for i in "$@"; do
    docker pull stellargraph/"$i":"$numbered_tag"
    docker tag stellargraph/"$i":"$numbered_tag" stellargraph/"$i":"$published_tag"
    docker push stellargraph/"$i":"$published_tag"
  done
}

delete_numbered() {
  local version=$1
  shift

  set +x # don't print out the docker hub credentials
  payload="{\"username\": \"${DOCKER_LOGIN_USER}\", \"password\": \"${DOCKER_LOGIN_PASSWORD}\"}"
  TOKEN=$(curl -s -H "Content-Type: application/json" -X POST -d "${payload}" https://hub.docker.com/v2/users/login/ | jq -r .token)
  authHeader="Authorization: JWT ${TOKEN}"

  for image_name in "$@"; do
    if [[ $version == *-${BUILDKITE_BUILD_NUMBER} ]]; then
      # This is a numbered tag, so we should try to delete it
      url="https://hub.docker.com/v2/repositories/stellargraph/${image_name}/tags/${version}"

      if curl --fail -L -X DELETE -H "$authHeader" "$url"; then
        echo "successfully deleted 'stellargraph/${image_name}:${version}'"
      else
        echo "failed to delete 'stellargraph/${image_name}:${version}' (probably never uploaded)"
      fi
    fi
  done
}

images=(
  stellargraph
)

action="$1"

case "${BUILDKITE_BRANCH}" in
  develop)
    version=latest
    ;;
  feature/*)
    version=testing
    ;;
esac

published_image_tag="${version}"
numbered_image_tag="${published_image_tag}-${BUILDKITE_BUILD_NUMBER}"

case "${action}" in
  build-publish-numbered)
    build_publish_numbered "${numbered_image_tag}" "${images[@]}"
    ;;
  publish-images)
    publish_images "${numbered_image_tag}" "${published_image_tag}" "${images[@]}"
    ;;
  delete-numbered)
    delete_numbered "${numbered_image_tag}" "${images[@]}"
    ;;
  *)
    echo "[ERROR] unknown argument '${action}'" >&2
    exit 1
    ;;
esac
