#!/bin/bash

set -xeo pipefail

junit_file="junit-${BUILDKITE_JOB_ID}.xml"

upload_tests() {
  buildkite-agent artifact upload "$junit_file" "s3://${AWS_LOGS_BUCKET}/pytest/${BUILDKITE_BRANCH}/${BUILDKITE_BUILD_NUMBER}"
}

echo "--- listing dependency versions"
pip freeze

exitCode=0

if [ "${CHECK_NOTEBOOK_FORMATTING-0}" = 1 ]; then
  echo "+++ checking notebook formatting"
  # This script takes only 20 seconds but requires non-trivial dependencies, so piggy back off the
  # installation that is happening in this CI step, rather than run it in a separate parallel step
  # where it would have to spend ~2 minutes installing dependencies.
  python scripts/format_notebooks.py --default --ci demos/ || exitCode=$?
fi

echo "+++ running tests"
py.test -ra --cov=stellargraph tests/ --doctest-modules --cov-report=xml -p no:cacheprovider --junitxml="./${junit_file}" || exitCode=$?

echo "--- :coverage::codecov::arrow_up: uploading coverage to codecov.io"
bash <(curl https://codecov.io/bash)

if [ "${BUILDKITE_BRANCH}" = "develop" ] || [ "${BUILDKITE_BRANCH}" = "master" ]; then
  echo "--- uploading JUnit"
  upload_tests
fi

# upload the JUnit file for the junit annotation plugin
buildkite-agent artifact upload "${junit_file}"

exit "$exitCode"
