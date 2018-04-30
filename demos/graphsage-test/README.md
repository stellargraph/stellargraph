# graphsage-test

Sandbox for testing graphsage implementations

## Running the example
Ensure you have a redis instance running locally on the default port 6379.
```
redis-server
```
Then run the example using:
```
python -m example.example_graphsage
```
Which writes the example PPI data to redis, then runs supervised graphsage in batches for 5 epochs.

The example data comes from the GraphSAGE implementation and example by the original authors. From their [README](https://github.com/williamleif/GraphSAGE):

> The example_data subdirectory contains a small example of the protein-protein interaction data, which includes 3 training graphs + one validation graph and one test graph. The full Reddit and PPI datasets (described in the paper) are available on the project website.

## References

```
 @inproceedings{hamilton2017inductive,
     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
     title = {Inductive Representation Learning on Large Graphs},
     booktitle = {NIPS},
     year = {2017}
   }
```