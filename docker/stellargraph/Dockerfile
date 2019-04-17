FROM python:3.6-slim as builder

ENV PATH=${PATH}:/root/.local/bin
WORKDIR /build/

COPY . /build/

RUN pip install --no-cache-dir -r /build/requirements.txt /build/ --user

FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar

COPY --chown=stellar --from=builder /root/.local /home/stellar/.local

USER stellar
WORKDIR /home/stellar
ENV PATH=${PATH}:/home/stellar/.local/bin
