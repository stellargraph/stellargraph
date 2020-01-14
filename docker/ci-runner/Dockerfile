ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim

RUN adduser --disabled-password --gecos '' stellar
USER stellar
ENV PATH=${PATH}:/home/stellar/.local/bin

WORKDIR /build
# Copy requirements first to install dependencies without having to recompute when the source code
# changes
COPY setup.py /build/setup.py
COPY stellargraph/version.py /build/stellargraph/version.py
# hadolint ignore=DL3013
RUN echo "+++ installing dependencies" \
    # install stellargraph without any source code to install its dependencies, and then immediately
    # uninstall it (without uninstalling the dependencies), so that the installation with source
    # code below will work without the `--upgrade` flag. This flag will cause pip to try to update
    # dependencies, which we don't want to happen in that second step.
    && pip install --no-cache-dir /build/ --user \
    && pip uninstall -y stellargraph
