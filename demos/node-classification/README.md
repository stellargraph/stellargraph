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

The dataset can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the cora.tgz file to a location on your computer and pass this location
as a command line argument to this script.

## Running the script

The example should be run on supplying the location of the downloaded CORA dataset with the following command:
```
python cora-example.py -g <path_to_cora_dataset>
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
