FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar

USER stellar
WORKDIR /home/stellar

ENV PATH=${PATH}:/home/stellar/.local/bin

COPY --chown=stellar . /home/stellar/stellargraph/

RUN pip install --upgrade pip --user \
      && pip install --no-cache-dir -r /home/stellar/stellargraph/requirements.txt /home/stellar/stellargraph/ --user \
      && rm -rf /home/stellar/stellargraph/
