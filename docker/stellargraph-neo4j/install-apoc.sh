#!/bin/bash

set -euo pipefail

echo "--- downloading apoc jar for neo4j ${NEO4J_VERSION}"

case "$NEO4J_VERSION" in
  4.0)
    apoc_version="4.0.0.4"
    apoc_sha="edea2796a9cd97f1daaeb291b5d88747fc4801dc63691eba6997157b34bd8db3525420a289e80d15f495c1d2fdbe00f9c626a74459f861db76e77c1c3e29e73b"
    ;;
  3.5)
    apoc_version="3.5.0.9"
    apoc_sha="b4b1a8f8940ad250c17b45296459afcfff2eb1beee18b8c8a394e2b5a434183b924b34cb5b04fea1a371fbfb87a286228bd4ae814829e0ef99f9e190a764fa1d"
    ;;
  *)
    echo "NEO4J_VERSION: unsupported version '${NEO4J_VERSION}'"
    ;;
esac

echo "APOC version ${apoc_version} (sha512 ${apoc_sha})"

wget --no-verbose "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${apoc_version}/apoc-${apoc_version}-all.jar" --directory-prefix plugins/

# validate the downloaded jar
echo "${apoc_sha}  ${PWD}/plugins/apoc-${apoc_version}-all.jar" | sha512sum -c
