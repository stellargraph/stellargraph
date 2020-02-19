# Link Prediction demo

The main.py script runs link prediction on a homogeneous or heterogeneous graph. When
the graph is heterogeneous, it can optionally be treated as homogeneous for representation learning; in
this case, the link prediction script can be thought of as a baseline for more advanced
algorithms that do not simplify the input graph.

In addition, for heterogeneous graphs, the link prediction script gives the user some control over what edge
types to predict including a choice of filtering these edges by one of their attributes. Currently, the script only
allows filtering by a date attribute in the format **dd/mm/yyyy**, e.g., *10/10/2005*. The edge attribute holding the date
can be given any text label in the graph, e.g., 'date', 'timestamp', 'start date', etc.

For example, given a heterogeneous network with nodes representing **people** and **products**, and links connecting people with products
(**purchased**) and people with people (**friend**), then a user can ask that links of type **friend** be predicted. In addition,
if links of type **friend** have a **date** property, for example, in the range *01/01/2000* to *01/01/2010*, then a user
can ask that links to be predicted should occur after *01/01/2005*. That is link data with a date before *01/01/2005* are
used for training, and data after that same date are put aside for predicting/testing.


## Requirements
This example assumes the `stellargraph` library and its requirements have been
installed by following the installation instructions in the README
of the library's [root directory](https://github.com/stellargraph/stellargraph).

### References

1. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.

2. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017

3. Social Computing Data Repository at ASU [http://socialcomputing.asu.edu]. R. Zafarani and H. Liu, (2009). Tempe, AZ: Arizona State University, School of Computing, Informatics and Decision Systems Engineering.
