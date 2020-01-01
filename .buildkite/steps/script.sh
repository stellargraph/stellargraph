#!/bin/bash

set -xeo pipefail

upload_tests() {
  buildkite-agent artifact upload "${BUILDKITE_BUILD_NUMBER}.xml" "s3://${AWS_LOGS_BUCKET}/pytest/${BUILDKITE_BRANCH}/${BUILDKITE_BUILD_NUMBER}"
}

echo "--- installing dependencies"
pip install -q --no-cache-dir -r requirements.txt -e .

echo "+++ running tests"
py.test -ra --cov=stellargraph tests/ --doctest-modules --doctest-modules --cov-report=term-missing -p no:cacheprovider --junitxml=./"${BUILDKITE_BUILD_NUMBER}".xml

echo "--- uploading coveralls"
coveralls

if [ "${BUILDKITE_BRANCH}" = "develop" ] || [ "${BUILDKITE_BRANCH}" = "master" ]; then
  echo "--- uploading JUnit"
  upload_tests
fi
