#!/bin/bash

set -xeo pipefail

pip install -q --no-cache-dir -r requirements.txt -e .
py.test -ra --cov=stellargraph tests/ --doctest-modules --doctest-modules --cov-report=term-missing -p no:cacheprovider --junitxml=./${BUILDKITE_BUILD_NUMBER}.xml
coveralls

# Upload junitxml to s3
buildkite-agent artifact upload "${BUILDKITE_BUILD_NUMBER}.xml" s3://stellargraph-logs-hosted/pytest/${BUILDKITE_BUILD_NUMBER}
