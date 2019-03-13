## Ensemble learning for graph neural network algorithms

This folder contains two [Jupyter](http://jupyter.org/) python notebooks demonstrating `StellarGraph` model calibration
for binary (`calibration-pubmed-link-prediction.ipynb`) and multi-class 
classification (`calibration-pubmed-node-classification.ipynb`) problems.


## Dataset

### Pubmed-Diabetes

The examples in this directory use the Pubmed-Diabetes dataset. 

The dataset can be downloaded from [here](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz).

The following is the description of the dataset:
> The Pubmed Diabetes dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes 
> classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset 
> is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.

Download and unzip the [Pubmed-Diabetes.tgz](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz) file to a 
location on your computer and set the corresponding variable in the notebook to point to this directory.

## References

**1.** On Calibration of Modern Neural Networks. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. ICML 
2017. ([link](https://geoffpleiss.com/nn_calibration))
