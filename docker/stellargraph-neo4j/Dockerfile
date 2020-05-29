ARG NEO4J_VERSION
FROM neo4j:${NEO4J_VERSION}

ARG NEO4J_VERSION

USER neo4j
COPY install-apoc.sh /
COPY install-gds.sh /
RUN bash /install-apoc.sh
RUN bash /install-gds.sh
