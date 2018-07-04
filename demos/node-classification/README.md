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

For the `epgm-example.py` script, a graph in EPGM format is required.

## Running the script

The example can be run on the cora graph (stored in EPGM format) with the following command:
```
python epgm-example.py -g ../../tests/resources/data/cora/cora.epgm -l 50 50 -s 20 10 -e 20 -d 0.5 -r 0.01
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
