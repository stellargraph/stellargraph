#!/bin/bash

set -euo pipefail

find . -name Dockerfile -o -name '*.dockerfile' | while IFS= read -r file; do
  echo "Checking formatting of $file"
  formatted="${file}.formatted"
  dockerfile-utils format --spaces 4 "$file" > "$formatted"

  if ! diff "$file" "$formatted"; then
    message="Dockerfile has incorrect formatting according to 'dockerfile-utils format --spaces 4 $file'. Formatted form:

$(cat "$formatted")"
    # replace the newlines with %0A for multiline output
    echo -n "::error file=${file}::$message" | awk 1 ORS="%0A"
    # separate from the next file's output
    echo
  fi
done
