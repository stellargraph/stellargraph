FROM python:3.6-slim as builder

RUN adduser --disabled-password --gecos '' stellar

USER stellar
WORKDIR /home/stellar

ENV PATH=${PATH}:/home/stellar/.local/bin

COPY --chown=stellar . /home/stellar/

RUN pip install --upgrade pip --user \
      && pip install --no-cache-dir -r /home/stellar/requirements.txt /home/stellar/ --user \
      && chown -R stellar .local/

FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar

USER stellar
WORKDIR /home/stellar
ENV PATH=${PATH}:/home/stellar/.local/bin

COPY --chown=stellar --from=builder /home/stellar/.local /home/stellar/.local
