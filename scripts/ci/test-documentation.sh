#!/bin/bash

set -euo pipefail

# need two files for the -w, because it overwrites not appends
error_file=/tmp/sphinx-errors.txt
spelling_error_file=/tmp/sphinx-spelling-errors.txt

spelling_file="_build/spelling/output.txt"
opts="-W --keep-going"

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
make html SPHINXOPTS="$opts -w $error_file" || exit_code="$?"

echo "+++ checking spelling"
make spelling SPHINXOPTS="$opts -w $spelling_error_file" || exit_code="$?"

if [ "$exit_code" -ne 0 ]; then
  echo "--- annotating build with failures"

  # strip out the /workdir/ references, so that the filenames are more relevant to the user
  # (relative to the repo root)
  output="$(cat "$error_file" "$spelling_error_file" | sed s@/workdir/@@)"
  if [ -s "$spelling_file" ]; then
    spelling="Mispelled words:

~~~terminal
$(cat "$spelling_file")
~~~

Any unusual words that are correctly spelled (for example, en-US spelling or appropriately-capitalised proper nouns) should be added to \`docs/spelling_wordlist.txt\`."
  else
    spelling=""
  fi

  exit 1
fi
