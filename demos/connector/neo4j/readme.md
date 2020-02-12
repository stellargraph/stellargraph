# Connect to Remote Neo4J Graph Databases

In this folder:
- [directed-graphsage-on-cora-neo4j-example.ipynb](./directed-graphsage-on-cora-neo4j-example.ipynb) and [undirected-graphsage-on-cora-neo4j-example.ipynb](./undirected-graphsage-on-cora-neo4j-example.ipynb) provides examples of running GraphSAGE with connection to graph database.

 - [load-cora-into-neo4j.ipynb](./load-cora-into-neo4j.ipynb) provides demo of loading Cora network citation dataset into neo4j database.

All the notebooks use ```py2neo```, a client library and toolkit to connect to Neo4J database from within python applications.

Install ```py2neo``` by using pip: `pip install py2neo`. Link to [`py2neo` documentation](https://py2neo.org/v4/).
