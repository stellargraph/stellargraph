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
      && curl -sLO "http://files.grouplens.org/datasets/movielens/ml-100k.zip" \
      && unzip ml-100k.zip \
      && curl -sLO "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip" \
      && unzip BlogCatalog-dataset.zip

# hadolint ignore=DL3006
FROM stellargraph/stellargraph

ENV JUPYTER_VERSION="1.0.0" \
  PATH=${PATH}:/home/stellar/.local/bin

# keep pip up to date for the end user
# hadolint ignore=DL3013
RUN pip install --upgrade pip --user \
      && pip install --no-cache-dir jupyter=="${JUPYTER_VERSION}" treon seaborn --user  

# python-igraph need to be installed for the community demo but there is no already built version of it for linux so can't be pip installed.
# A lot of dependencies need to be installed.

COPY --chown=stellar ./demos /home/stellar/demos
COPY --chown=stellar --from=base /data/cora /home/stellar/data/cora
COPY --chown=stellar --from=base /data/Pubmed-Diabetes /home/stellar/data/pubmed/Pubmed-Diabetes
COPY --chown=stellar --from=base /data/ml-100k /home/stellar/data/ml-100k
COPY --chown=stellar --from=base /data/BlogCatalog-dataset /home/stellar/data/BlogCatalog-dataset
COPY --chown=stellar ./scripts/ /home/stellar/scripts


EXPOSE 8888

CMD ["python",  "/home/stellar/scripts/test_demos.py"]
