FROM python:3.8-slim

RUN pip3 --no-cache-dir install black==19.10b

ENTRYPOINT ["black", "--check", "--diff", "."]
