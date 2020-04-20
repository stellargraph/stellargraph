#!/bin/sh

# get all notebooks
for nb_path in $(find demos -name "*.ipynb"); do
  nb_link="docs/${nb_path%.ipynb}.nblink"

  # recurse through parent directories and create index.txt
  nb_link_dir="$(dirname "$nb_link")"
  while [ "$nb_link_dir" != "docs" ]; do
    nb_link_parent_index="$nb_link_dir/index.txt"
    if [[ ! -f "$nb_link_parent_index" ]]; then
      echo "Creating '$nb_link_parent_index'"
      mkdir -p "$nb_link_dir"
      cat > "$nb_link_parent_index" << EOF
$(basename "$nb_link_dir")
====================================

.. toctree::
    :titlesonly:
    :glob:

    */index
    ./*
EOF
    else
      break
    fi
    nb_link_dir="$(dirname "$nb_link_dir")"
  done

  # create .nblink file
  echo "Creating '$nb_link'"
  nb_link_dir="$(dirname "$nb_link")"
  link_relative_path="$(realpath --relative-to="$nb_link_dir" "$nb_path")"
  cat > "$nb_link" << EOF
{
  "path": "$link_relative_path"
}
EOF
done
