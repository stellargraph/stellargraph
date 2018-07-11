#!/bin/bash

curl -s --output /dev/null --show-error --fail -X POST -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" \
  https://api.buildkite.com/v2/organizations/stellar/pipelines/buildkite-logger/builds -d \
  '{"message":"trigger build",
  "commit":"HEAD",
  "branch":"master",
  "env":{
  "LOGGER_PIPELINE_SLUG":"'"${BUILDKITE_PIPELINE_SLUG}"'",
  "LOGGER_BRANCH":"'"${BUILDKITE_BRANCH}"'",
  "LOGGER_BUILD_NUMBER":"'"${BUILDKITE_BUILD_NUMBER}"'",
  "LOGGER_SLACK_CHANNEL":"'"$1"'"
}
}'
