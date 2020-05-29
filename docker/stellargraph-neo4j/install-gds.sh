#!/bin/bash

set -euo pipefail

echo "--- downloading gds jar for neo4j ${NEO4J_VERSION}"

case "$NEO4J_VERSION" in
  4.0)
    gds_version="1.2.1"
    ;;
  3.5)
    gds_version="1.2.1"
    ;;
  *)
    echo "NEO4J_VERSION: unsupported version '${NEO4J_VERSION}'"
    ;;
esac

wget --no-verbose "https://github.com/neo4j/graph-data-science/releases/download/${gds_version}/neo4j-graph-data-science-${gds_version}-standalone.jar" --directory-prefix plugins/
echo 'dbms.security.procedures.unrestricted=apoc.*,gds.*' >> ${PWD}/conf/neo4j.conf

# validate the downloaded jar
# echo "${apoc_sha}  ${PWD}/plugins/apoc-${apoc_version}-all.jar" | sha512sum -c
