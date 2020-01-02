FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar
USER stellar
WORKDIR /home/stellar

COPY . /build/

# installs the packages to /root/.local
RUN pip install --no-cache-dir -r /build/requirements.txt /build/ --user

ENV PATH=${PATH}:/home/stellar/.local/bin
