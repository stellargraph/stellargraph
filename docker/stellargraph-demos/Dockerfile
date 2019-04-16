FROM alpine:3.9 as base

WORKDIR /data

# Download datasets
RUN apk add --no-cache --update ca-certificates \
      && wget -q "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz" \
      && tar --no-same-owner -xf cora.tgz \
      && wget -q "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz" \
      && tar --no-same-owner -xf Pubmed-Diabetes.tgz \
      && wget -q "http://files.grouplens.org/datasets/movielens/ml-100k.zip" \
      && unzip ml-100k.zip \
      && wget -q "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip" \
      && unzip BlogCatalog-dataset.zip

FROM stellargraph/stellargraph

ENV JUPYTER_VERSION="1.0.0"

RUN pip install --upgrade pip --user \
      && pip install jupyter=="${JUPYTER_VERSION}" --user

COPY --chown=stellar ./demos /home/stellar/demos
COPY --chown=stellar --from=base /data/cora /home/stellar/data/cora
COPY --chown=stellar --from=base /data/Pubmed-Diabetes /home/stellar/data/pubmed/Pubmed-Diabetes
COPY --chown=stellar --from=base /data/ml-100k /home/stellar/data/ml-100k
COPY --chown=stellar --from=base /data/BlogCatalog-dataset /home/stellar/data/BlogCatalog-dataset

EXPOSE 8888

CMD ["jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", "--notebook-dir=/home/stellar", "--NotebookApp.token=''","--NotebookApp.password=''"]
