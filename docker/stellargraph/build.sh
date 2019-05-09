#!/bin/bash

VERSION="latest"
BASEDIR=$(dirname "$0")
IMAGENAME="stellargraph/stellargraph"

docker build -t ${IMAGENAME}:"${VERSION}" -f "${BASEDIR}"/Dockerfile "${BASEDIR}/../../"
