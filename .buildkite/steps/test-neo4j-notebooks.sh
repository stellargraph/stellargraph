#!/bin/bash

set -xeo pipefail

echo "--- :python: installing stellargraph"
# install stellargraph itself and the dependencies necessary for neo4j, which (hopefully) won't
# install any other dependencies because they exist in the docker image already. (The neo4j deps
# aren't installed into the docker image, because we want the main tests to run without them,
# validating they're properly optional.)
pip install .[neo4j]

echo "--- listing dependency versions"
pip freeze

directory="$PWD/demos/connector/neo4j"
notebooks=(
  "../../basics/loading-saving-neo4j.ipynb"
  "load-cora-into-neo4j.ipynb"
  "directed-graphsage-on-cora-neo4j-example.ipynb"
  "undirected-graphsage-on-cora-neo4j-example.ipynb"
  "cluster-gcn-on-cora-neo4j-example.ipynb"
)

# delay to wait for Neo4j startup with plugins (fix for #1648)
sleep 10s

for name in "${notebooks[@]}"; do
  .buildkite/steps/test-single-notebook.sh "$directory/$name" " using Neo4j ${NEO4J_VERSION}"
done
