ARG NEO4J_VERSION
FROM neo4j:${NEO4J_VERSION}

ARG NEO4J_VERSION

USER neo4j
COPY install-lib.sh /
RUN bash /install-lib.sh
