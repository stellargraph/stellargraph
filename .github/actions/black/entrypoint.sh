#!/bin/sh

# Adapted from https://github.com/reviewdog/action-template

set -e

if [ -n "${GITHUB_WORKSPACE}" ] ; then
  cd "${GITHUB_WORKSPACE}/${INPUT_WORKDIR}" || exit
fi

export REVIEWDOG_GITHUB_API_TOKEN="${INPUT_GITHUB_TOKEN}"

# diff for more details, reviewdog check for
if ! black --diff --check . ; then
  version="$(black --version)"
  black --check . \
    | sed "s/would reformat \(.*\)/\1: invalid formatting according to $version/" \
    | reviewdog \
        -efm="%f: %m" \
        -name="black" \
        -reporter="${INPUT_REPORTER:-github-pr-check}" \
        -filter-mode="${INPUT_FILTER_MODE}" \
        -fail-on-error="${INPUT_FAIL_ON_ERROR}" \
        -level="${INPUT_LEVEL}" \
        ${INPUT_REVIEWDOG_FLAGS}
fi

