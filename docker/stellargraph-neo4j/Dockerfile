FROM neo4j:3.5

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV APOC_VERSION="3.5.0.9" APOC_SHA="7f630b1ba3dccd0b59fa609a3f488820059044b3075d9604a79741fea379544c"
RUN echo "--- downloading apoc jar" && \
    wget --no-verbose "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar" --directory-prefix plugins/ && \
    echo "${APOC_SHA}  ${PWD}/plugins/apoc-${APOC_VERSION}-all.jar" | sha256sum -c
