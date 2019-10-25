#!/bin/bash

VERSION="latest"
BASEDIR=$(dirname "$0")
IMAGENAME="stellargraph/stellargraph-treon"

docker build -t ${IMAGENAME}:"${VERSION}" -f "${BASEDIR}"/Dockerfile "${BASEDIR}/../../"
