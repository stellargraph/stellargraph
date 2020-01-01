#!/bin/bash

set -xeo pipefail

junit_file="junit/${BUILDKITE_JOB_ID}.xml"

upload_tests() {
  buildkite-agent artifact upload "$junit_file" "s3://${AWS_LOGS_BUCKET}/pytest/${BUILDKITE_BRANCH}/${BUILDKITE_BUILD_NUMBER}"
}

pip install -q --no-cache-dir -r requirements.txt -e .
py.test -ra --cov=stellargraph tests/ --doctest-modules --doctest-modules --cov-report=term-missing -p no:cacheprovider --junitxml="./${junit_file}"
coveralls

if [ "${BUILDKITE_BRANCH}" = "develop" ] || [ "${BUILDKITE_BRANCH}" = "master" ]; then
  upload_tests
fi

# upload the JUnit file for the junit annotation plugin
buildkite-agent artifact upload "${junit_file}"
