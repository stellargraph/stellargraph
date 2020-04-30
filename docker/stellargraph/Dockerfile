FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar
# If WORKDIR creates the directory, it's owned by root, but it needs to be owned by stellar
RUN mkdir /build && chown stellar:stellar /build
USER stellar
ENV PATH=${PATH}:/home/stellar/.local/bin

WORKDIR /build
# Copy requirements first to install dependencies without having to recompute when the source code
# changes
COPY --chown=stellar:stellar setup.py /build/setup.py
COPY --chown=stellar:stellar stellargraph/version.py /build/stellargraph/version.py
# hadolint ignore=DL3013
RUN echo "+++ installing dependencies" \
    # install stellargraph without any source code to install its dependencies, and then immediately
    # uninstall it (without uninstalling the dependencies), so that the installation with source
    # code below will work without the `--upgrade` flag. This flag will cause pip to try to update
    # dependencies, which we don't want to happen in that second step.
    && pip install --no-cache-dir /build/ --user \
    && pip uninstall -y stellargraph

# Now copy the code in, to install stellargraph itself
COPY --chown=stellar:stellar . /build/
# hadolint ignore=DL3013
RUN echo "+++ installing stellargraph" && pip install --no-cache-dir /build/ --user
WORKDIR /home/stellar
