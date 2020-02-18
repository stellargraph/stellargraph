# Link Prediction demo

This demo runs link prediction on a homogeneous or heterogeneous graph. When
the graph is heterogeneous, it can optionally be treated as homogeneous for representation learning; in
this case, the link prediction script can be thought of as a baseline for more advanced
algorithms that do not simplify the input graph.

## Requirements
This example assumes the `stellargraph` library and its requirements have been
installed by following the installation instructions in the README
of the library's [root directory](https://github.com/stellargraph/stellargraph).

### References

1. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.

2. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017

3. Social Computing Data Repository at ASU [http://socialcomputing.asu.edu]. R. Zafarani and H. Liu, (2009). Tempe, AZ: Arizona State University, School of Computing, Informatics and Decision Systems Engineering.
