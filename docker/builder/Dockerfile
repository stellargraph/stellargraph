ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim


RUN echo "--- installing packages" \
    && apt update && apt install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' stellar
USER stellar
WORKDIR /home/stellar
ENV PATH=${PATH}:/home/stellar/.local/bin

COPY requirements.txt /build/requirements.txt
RUN echo "--- installing Python dependencies" \
    && pip install --no-cache-dir -r /build/requirements.txt --user
