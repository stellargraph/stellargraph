## Ensemble learning for graph neural network algorithms

This folder contains two [Jupyter](http://jupyter.org/) python notebooks demonstrating the use of ensemble learning
for node attribute inference (`ensemble-node-classification-example.ipynb`) and 
link prediction (`ensemble-link-prediction-example.ipynb`) using `StellarGraph`'s graph neural network algorithms.


## Datasets

### Cora

Some of the examples in this directory uses the CORA dataset. 

The dataset can be downloaded from [here](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz).

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a 
location on your computer and set the corresponding variable in the notebook to point to this directory.

### Pubmed-Diabetes

Some of the examples in this directory uses the Pubmed-Diabetes dataset. 

The dataset can be downloaded from [here](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz).

The following is the description of the dataset:
> The Pubmed Diabetes dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes 
> classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset 
> is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.

Download and unzip the [Pubmed-Diabetes.tgz](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz) file to a 
location on your computer and set the corresponding variable in the notebook to point to this directory.