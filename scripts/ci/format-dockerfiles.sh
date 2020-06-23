#!/bin/bash

set -euo pipefail

find . -name Dockerfile -o -name '*.dockerfile' | while IFS= read -r file; do
  echo "--- Checking formatting of $file"
  formatted="${file}.formatted"
  diff="${file}.diff"
  dockerfile-utils format --spaces 4 "$file" > "$formatted"

  if ! diff "$file" "$formatted"; then
    message="Dockerfile has incorrect formatting according to 'dockerfile-utils format --spaces 4 $file'. Formatted form:

$formatted"
    # replace the newlines with %0A for multiline output
    echo "::error file=${file}::$message" | awk 1 ORS="%0A"
  fi
done

