FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar
USER stellar
WORKDIR /home/stellar

COPY requirements.txt /build/requirements.txt
RUN echo "+++ installing dependencies" && pip install --no-cache-dir -r /build/requirements.txt --user

COPY . /build/
RUN echo "+++ installing stellargraph" && pip install --no-cache-dir /build --user
ENV PATH=${PATH}:/home/stellar/.local/bin
