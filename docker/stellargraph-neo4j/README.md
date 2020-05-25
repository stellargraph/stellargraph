# Neo4j Docker image

This image is being used in CI for testing our Neo4j functionality, but it can also be used for local development.

You can build the image for Neo4J 4.0 using the build script:
```
./build.sh
```

Run the docker container:
```
docker run -it -e NEO4J_AUTH=none -p 7687:7687 -p 7474:7474 stellargraph_neo4j:latest
```

