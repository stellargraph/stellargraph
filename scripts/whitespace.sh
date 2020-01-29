#!/bin/bash

set -euo pipefail

usage="$(basename "$0") [-h | --help] [--ci] -- script to normalise whitespace, or validate that it is normalised on CI"

on_ci=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h | --help)
      echo "$usage"
      exit 0
      ;;
    --ci)
      on_ci=1
      shift
      ;;
    *)
      echo "$usage"
      echo
      echo "unknown option: $1"
      exit 1
      ;;
  esac
done

# for every (git ls-files) non-binary (grep -I) file, normalise the whitespace and write it back, to
# be able to do a `git diff` to get nice output
git ls-files -z | xargs -0 grep --null -I --files-with-match '' | while read -rd $'\0' file; do
  # strip trailing whitespace from each line with sed, and use bash's collapsing to delete multiple
  # newlines
  contents=$(sed 's/  *$//' "$file")
  # overwrite the file with the good-whitespace version (this will also ensure there's a newline at
  # the end, if there wasn't one originally)
  printf "%s\n" "$contents" > "$file"
done

if [ "$on_ci" -eq 1 ]; then
  tmp="$(mktemp)"
  if git diff --exit-code --color=always --ws-error-highlight=all > "$tmp"; then
    echo ":tada: All files have no trailing whitespace and exactly one newline at the end."
  else
    # we want literal `s, not command subtitution:
    # shellcheck disable=SC2016
    msg='Found files with trailing whitespace or not exactly one newline at the end (run `./scripts/whitespace.sh` to fix):'

    echo "$msg"
    echo
    # diff output includes --- and +++ which are meaningful to buildkite at the start of the line
    # (https://buildkite.com/docs/pipelines/managing-log-output) so indent the diff lines a bit
    sed 's/^/  /' "$tmp"

    buildkite-agent annotate --context whitespace --style error << EOF
$msg

<details>

\`\`\`term
$(cat "$tmp")
\`\`\`

</details>
EOF

    exit 1
  fi
fi
