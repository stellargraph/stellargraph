# Image is needed anyway, so it's also used for downloading datasets
FROM python:3.6-slim as base

WORKDIR /data

# Download datasets
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl unzip \
    && curl -sLO "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz" \
    && tar --no-same-owner -xf cora.tgz \
    && curl -sLO "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz" \
    && tar --no-same-owner -xf Pubmed-Diabetes.tgz \
    && curl -sLO "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip" \
    && unzip BlogCatalog-dataset.zip

# hadolint ignore=DL3006
FROM stellargraph/stellargraph

ENV PATH=${PATH}:/home/stellar/.local/bin \
    JUPYTER_VERSION="1.0.0" \
    TREON_VERSION="0.1.2" \
    SEABORN_VERSION="0.9.0"


# keep pip up to date for the end user
# hadolint ignore=DL3013
RUN pip install --upgrade pip --user \
    && pip install --no-cache-dir jupyter=="${JUPYTER_VERSION}" treon=="${TREON_VERSION}" seaborn=="${SEABORN_VERSION}" --user

COPY --chown=stellar ./demos /home/stellar/demos
COPY --chown=stellar ./scripts/ /home/stellar/scripts
COPY --chown=stellar --from=base /data/cora /home/stellar/data/cora
COPY --chown=stellar --from=base /data/Pubmed-Diabetes /home/stellar/data/pubmed/Pubmed-Diabetes
COPY --chown=stellar --from=base /data/BlogCatalog-dataset /home/stellar/data/BlogCatalog-dataset

CMD ["sh", "-c", "python /home/stellar/scripts/test_demos.py"]
