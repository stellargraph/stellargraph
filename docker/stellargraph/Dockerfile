FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar
USER stellar
WORKDIR /home/stellar
ENV PATH=${PATH}:/home/stellar/.local/bin

COPY requirements.txt /build/requirements.txt
RUN echo "+++ installing dependencies" && pip install --no-cache-dir -r /build/requirements.txt --user

COPY . /build/
# hadolint ignore=DL3013
RUN echo "+++ installing stellargraph" && pip install --no-cache-dir /build/ --user
