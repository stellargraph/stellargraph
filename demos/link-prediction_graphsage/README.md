# Graphsage Link Prediction

This is an example of using Homogenous GraphSAGE model, with a link classifier on top, 
to predict links in a graph.
The link prediction problem is treated as a supervised binary classification problem for 
`(src, dst)` node pairs that make up links in the graph, with positive examples
representing links that do exist in the graph, and negative examples representing
links that don't. 

In this example, we learn to predict citation links between papers in a Cora dataset (see below).

## Requirements
Install the Stellar ML library.

In top-level stellar-ml directory install using pip:

```
pip install -e .
```

## CORA dataset

Currently this example is tested on the CORA dataset. The GraphSAGE model assumes that node
features are available.

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a location on your computer and pass this location
as a command line argument to this script.

## Running the script

The example should be run on supplying the location of the downloaded CORA dataset with the following command:
```
python cora-links-example.py -g <path_to_cora_dataset>
```
The above command runs the link prediction on Cora dataset with default
parameters. For help on how to set parameters of the run and their meaning, run
```angular2html
python cora-links-example.py -h
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
