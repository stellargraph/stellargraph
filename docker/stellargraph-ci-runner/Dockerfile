ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}

ARG PRERELEASE_VERSIONS

WORKDIR /build
# Copy requirements first to install dependencies without having to recompute when the source code
# changes
COPY setup.py /build/setup.py
COPY stellargraph/version.py /build/stellargraph/version.py
COPY docker/stellargraph-ci-runner/install-packages.sh /install-packages.sh

RUN bash /install-packages.sh
