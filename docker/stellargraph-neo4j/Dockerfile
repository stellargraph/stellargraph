FROM neo4j:3.5

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

USER neo4j

ENV APOC_VERSION="3.5.0.9" APOC_SHA="b4b1a8f8940ad250c17b45296459afcfff2eb1beee18b8c8a394e2b5a434183b924b34cb5b04fea1a371fbfb87a286228bd4ae814829e0ef99f9e190a764fa1d"
RUN echo "--- downloading apoc jar" && \
    wget --no-verbose "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar" --directory-prefix plugins/ && \
    echo "${APOC_SHA}  ${PWD}/plugins/apoc-${APOC_VERSION}-all.jar" | sha512sum -c
