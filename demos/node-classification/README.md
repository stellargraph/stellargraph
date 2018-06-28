# Graphsage Node Classification

This is an example of using Homogenous GraphSAGE to classify nodes in a graph.

## Requirements
Install the Stellar ML library.

In top-level stellar-ml directory install using pip:

```
pip install -e .
```

## CORA dataset

Currently this example is tested on the CORA dataset. The GraphSAGE model assumes that node
features are available.

### Getting the data

For the `epgm-example.py` script, a graph in EPGM format is required.

## Running the script

Run the example for ML-1m with movie & user features using the following command:
```
python epgm-example.py -g ../../tests/resources/data/cora/cora.epgm -n 100 -e 20 -l 50 20 -s 20 10
```

## References

```
 @inproceedings{hamilton2017inductive,
     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
     title = {Inductive Representation Learning on Large Graphs},
     booktitle = {NIPS},
     year = {2017}
   }
```
