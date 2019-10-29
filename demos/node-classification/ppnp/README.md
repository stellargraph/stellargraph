# PPNP and APPNP for Node Classification

This directory contains examples of using the Personalized Propogation of Neural Predictions (PPNP) and Approximate PPNP (APPNP)
algorithms [1] for semi-supervised node classification in a homogeneous network.

## Requirements
All examples use Python 3.6 and the StellarGraph library. To install the StellarGraph library
follow the instructions at: https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy, Keras, and Scikit-Learn which are installed as depdendencies
of the StellarGraph library. 

## CORA dataset

The CORA dataset can be downloaded from [here](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)).

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a location on your computer.

## Reddit dataset

The reddit dataset can be downloaded from [here](http://snap.stanford.edu/graphsage/).

The following is a description of the dataset [2]:

>Reddit is a large online discussion forum where users post and comment on content in different topical
>communities. We constructed a graph dataset from Reddit posts made in the month of September, 2014. The node label in this case is the community, or “subreddit”, that a post belongs to. We sampled
>50 large communities and built a post-to-post graph, connecting posts if the same user comments
>on both. In total this dataset contains 232,965 posts with an average degree of 492. We use the first
>20 days for training and the remaining days for testing (with 30% used for validation). For features,
>we use off-the-shelf 300-dimensional GloVe CommonCrawl word vectors [3]; for each post, we
>concatenated (i) the average embedding of the post title, (ii) the average embedding of all the post’s
>comments (iii) the post’s score, and (iv) the number of comments made on the post.


## References

[1] Predict then propagate: Graph neural networks meet personalized pagerank. J. Klicpera,  A. Bojchevski, & S. Günnemann arxiv:1810.05997, 2018.


[2] Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216 [cs.SI], 2017.


[3] Glove: Global vectors for word representation. J. Pennington, R. Socher, and C. D. Manning. In EMNLP, 2014.
