#!/bin/bash

set -xeo pipefail

junit_file="junit-${BUILDKITE_JOB_ID}.xml"

upload_tests() {
  buildkite-agent artifact upload "$junit_file" "s3://${AWS_LOGS_BUCKET}/pytest/${BUILDKITE_BRANCH}/${BUILDKITE_BUILD_NUMBER}"
}

echo "--- listing dependency versions"
pip freeze

exitCode=0

echo "+++ running tests"
# benchmarks on shared infrastructure like the CI machines are usually unreliable (high variance),
# so there's no point spending too much time. It's good to check that the benchmark does run
# successfully as a benchmark (that is, multiple iterations, not just once, as a test with
# --benchmark-disable)
small_benchmarks=(--benchmark-max-time=0.01 --benchmark-warmup=off --benchmark-min-rounds=2)

py.test -ra --cov=stellargraph tests/ --doctest-modules --cov-report=xml -p no:cacheprovider --junitxml="./${junit_file}" "${small_benchmarks[@]}" || exitCode=$?

echo "--- :coverage::codecov::arrow_up: uploading coverage to codecov.io"
bash <(curl https://codecov.io/bash)

if [ "${BUILDKITE_BRANCH}" = "develop" ] || [ "${BUILDKITE_BRANCH}" = "master" ]; then
  echo "--- uploading JUnit"
  upload_tests
fi

# upload the JUnit file for the junit annotation plugin
buildkite-agent artifact upload "${junit_file}"

exit "$exitCode"
