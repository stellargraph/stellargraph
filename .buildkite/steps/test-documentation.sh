#!/bin/bash

set -euo pipefail

error_file=/tmp/sphinx-errors.txt

spelling_file="_build/spelling/output.txt"
opts="-W --keep-going -w $error_file"

cd docs

echo "--- installing pandoc"
apt update
apt install -y pandoc enchant

echo "--- installing documentation requirements"
pip install -r requirements.txt

echo "--- listing dependency versions"
pip freeze

exit_code=0
echo "+++ building docs"
make html SPHINXOPTS="$opts" || exit_code="$?"

echo "+++ checking spelling"
make spelling SPHINXOPTS="$opts" || exit_code="$?"

if [ "$exit_code" -ne 0 ]; then
  echo "--- annotating build with failures"

  # strip out the /workdir/ references, so that the filenames are more relevant to the user
  # (relative to the repo root)
  output="$(sed s@/workdir/@@ "$error_file")"
  if [ -s "$spelling_file" ]; then
    spelling="Mispelled words:

~~~terminal
$(cat "$spelling_file")
~~~

Any unusual words that are correctly spelled (for example, appropriately-capitalised proper nouns) should be added to \`docs/spelling_wordlist.txt\`."
  else
    spelling=""
  fi

  buildkite-agent annotate --context "sphinx-doc-build" --style error << EOF
The sphinx build had warnings and/or errors. These may mean that the documentation doesn't display as expected and so should be fixed.

~~~terminal
$output
~~~

$spelling

[View all output](#$BUILDKITE_JOB_ID)
EOF
  exit 1
fi
