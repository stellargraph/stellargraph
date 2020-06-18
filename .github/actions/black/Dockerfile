FROM python:3.8-slim

ENV REVIEWDOG_VERSION=v0.10.0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# hadolint ignore=DL3006
RUN apk --no-cache add git

RUN wget -O - -q https://raw.githubusercontent.com/reviewdog/reviewdog/master/install.sh| sh -s -- -b /usr/local/bin/ ${REVIEWDOG_VERSION}

RUN pip3 --no-cache-dir install black==19.10b

ENTRYPOINT ["black", "--check", "--diff", "."]
