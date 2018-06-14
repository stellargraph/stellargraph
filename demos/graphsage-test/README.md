# graphsage-test

Sandbox for testing graphsage implementations

## Running the examples
Prior to running any of the examples, ensure you have a redis instance running locally on the default port 6379.
```
redis-server
```

### PPI
The PPI example data comes from the GraphSAGE implementation and example by the original authors. From their [README](https://github.com/williamleif/GraphSAGE):
> The example_data subdirectory contains a small example of the protein-protein interaction data, which includes 3 training graphs + one validation graph and one test graph. The full Reddit and PPI datasets (described in the paper) are available on the project website.

Run the example using:
```
python -m example.example_graphsage
```
Which writes the example PPI data to redis, then runs supervised graphsage in batches for 5 epochs.

### MovieLens
The MovieLens data contains a bipartite graph of Movies and Users, and edges between them denoting reviews with scores 
ranging from 1 to 5. 

Run the example using:
```
python -m example.movielens
```
The example trains HinSAGE to predict the "score" attribute on links. This example runs for 10 epochs and may run for 
around 1 hour on a laptop.

## References

```
 @inproceedings{hamilton2017inductive,
     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
     title = {Inductive Representation Learning on Large Graphs},
     booktitle = {NIPS},
     year = {2017}
   }
```