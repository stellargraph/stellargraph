# Connect to Remote Neo4J Graph Databases

This folder contains notebook demos of running Stellargraph ML algorithms with a graph database.

[directed-graphsage-on-cora-neo4j-example.ipynb](./directed-graphsage-on-cora-neo4j-example.ipynb) and [undirected-graphsage-on-cora-neo4j-example.ipynb](./undirected-graphsage-on-cora-neo4j-example.ipynb) use ```py2neo```, a client library and toolkit to connect to Neo4J database from within python applications.

Install ```py2neo``` by using pip: `pip install py2neo`. Link to [`py2neo` documentation](https://py2neo.org/v4/).

[load-cora-into-neo4j.ipynb](./load-cora-into-neo4j.ipynb) provides demo of loading Cora network citation dataset into neo4j database.
