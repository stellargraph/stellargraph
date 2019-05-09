#!/bin/bash

BUILDKITE_BUILD_STATE="$(curl -s --show-error --fail -X GET -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" "https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds/${BUILDKITE_BUILD_NUMBER}" | jq -r ".jobs[].state" | sort -u | grep "failed\|canceled")"

case "${BUILDKITE_BUILD_STATE}" in
  failed|canceled)
    echo "build ${BUILDKITE_BUILD_STATE}: not running $*"
    ;;
  *)
    exec "$@"
    ;;
esac
