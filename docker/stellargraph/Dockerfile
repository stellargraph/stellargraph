FROM python:3.6-slim

RUN adduser --disabled-password --gecos '' stellar
USER stellar
WORKDIR /home/stellar
ENV PATH=${PATH}:/home/stellar/.local/bin

# Copy requirements first to install dependencies without having to recompute when the source code
# changes
COPY requirements.txt /build/requirements.txt
RUN echo "+++ installing dependencies" && pip install --no-cache-dir -r /build/requirements.txt --user

# Now copy the code in, to install stellargraph itself
COPY . /build/
# hadolint ignore=DL3013
RUN echo "+++ installing stellargraph" && pip install --no-cache-dir /build/ --user
