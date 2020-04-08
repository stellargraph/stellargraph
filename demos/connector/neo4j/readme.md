# Connect to Remote Neo4J Graph Databases

In this folder:
- [directed-graphsage-on-cora-neo4j-example.ipynb](./directed-graphsage-on-cora-neo4j-example.ipynb) and [undirected-graphsage-on-cora-neo4j-example.ipynb](./undirected-graphsage-on-cora-neo4j-example.ipynb) provide examples of running GraphSAGE with connection to graph database.

- [load-cora-into-neo4j.ipynb](./load-cora-into-neo4j.ipynb) provides demo of loading Cora network citation dataset into neo4j database.

Required Installations:

- **Neo4J**: Instruction to download [here](https://neo4j.com/docs/operations-manual/current/installation/).
- **APOC library** plug-in: APOC provides utilities for common procedures and functions in Neo4J. Instruction to download and install [here](https://neo4j.com/developer/neo4j-apoc/).
- **py2neo**: A client library and toolkit to connect to Neo4J database from within python applications. Install ```py2neo``` by using pip: `pip install py2neo`. `py2neo` documentation [here](https://py2neo.org/v4/).


> :warning: All functionalities demonstrated in the above-mentioned notebooks are experimental. They have not been tested thoroughly and the implementation might be dramatically changed.

There is also [a demonstration of loading data into memory from Neo4j](../../basics/loading-saving-neo4j.ipynb).  This allows using any StellarGraph algorithm on data from Neo4j.
