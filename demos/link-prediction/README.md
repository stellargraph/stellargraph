# Link Prediction demo

The main.py script runs link prediction on a homogeneous or heterogeneous graph. When
the graph is heterogeneous, it can optionally be treated as homogeneous for representation learning; in
this case, the link prediction script can be thought of us a baseline for more advanced
algorithms that do not simplify the input graph. 

In addition, for heterogeneous graphs, the link prediction script gives the user some control over what edge 
types to predict including a choice of filtering these edges by one of their attributes. Currently, the script only
allows filtering by a date attribute in the format **dd/mm/yyyy**, e.g., *10/10/2005*. The edge attribute holding the date
can be given any text label in the graph, e.g., 'date', 'timestamp', 'start date', etc. 

For example, given a heterogeneous network of with **people**, **products**, and links connecting people with products 
(**purchased**) and people with people (**friend**), then a user can ask that links of type **friend** be predicted. In addition,
if edges of type **friend** have a **date** property, for example, in the range *01/01/2000* to *01/01/2010*, then a user
can ask that edges to be predicted should occur after *01/01/2005*. That is edge data with a date before *01/01/2005* are
used for training, and data after that same date are put aside for predicting.


### Instalation
To install the requirements for running the script, create a new python v3.6 environment
and execute,

pip install -r requirements.txt

### Usage

#### Command Line Arguments
The script accepts the following command line arguments:
               
- `input_graph <str>`  The directory where the graph in EPGM format is stored.
- `output_node_features <str>` The file where the node features from representation learning are written 
for future reference.
- `subsample` If specified, the graph is subsampled (number of nodes reduced) by a default 0.1 factor, e.g,
10 percent of the original graph. This option is useful for speeding up testing on large graphs.
- `subgraph_size <float>` Valid values are in the interval (0, 1]. It reduces the graph size by the given factor. 
Size reduction is performed by reducing the number of nodes in the graph by the given factor and removing all other
nodes and edges connecting those nodes.
- `sampling_method <str>` Valid values are 'local' and 'global'. Specifies how pairs of nodes that are not connected are 
selected as negative edge samples. The 'global' method, selects pairs of nodes uniformly at random from all nodes in 
the graph that are not connected by an edge. The 'local' methods first samples a distance (number of edges) from a 
source node (selected uniformly at random from all nodes int he graph) and then selects the first node at this distance 
as the target node for the negative edge sample. The distance is sampled based on a probability distribution that can 
be specified using the probs command line parameter.
- `sampling_probs <str: comma separated list of floats>` The probability distribution for sampling distances between 
source and target nodes. The first value should always be 0.0 and the remaining values should sum to one. An example for 
sampling target nodes up to 4 edges away from source is `"0.0, 0.25, 0.25, 0.5"`.
- `p <float>` If specified, it indicates the number of positive and negative edges to be used for training
the link prediction classifier. The value indicates a percentage with respect to the relevant number of
edges in the input graph. The default value is 0.1 and valid values are in the range (0,1).
- `hin` If specified, it indicates that the input graph is heterogeneous. If not specified, a heterogeneous graph is
simplified to homogeneous and the options `edge_type`, `edge_attribute_label`, `edge_attribute_threshold`, and
`attribute_is_datetime` are ignored even if given.
- `metapaths <str>` For heterogeneous graphs (must specify `hin` option), this option can be used to specify the 
metapaths for the random walks. A metapath is a `,` separated list of node labels; more than one metapaths can
be specified separated by a `;`. An example specifying 2 metapaths assuming node labels `author, paper, venue` is
`"author, paper, author; author, paper, venue, paper, author"`.
- `edge_type <str>` For heterogeneous graphs, this option is used to specify the type of edge (by its label) to
predict. 
- `edge_attribute_label <str>` For heterogeneous graphs, this option is used to specify the edges that should be
predicted based on an attribute; the only valid attribute is a date in the format dd/mm/yyyy, e.g., 10/2/2018. This
option should be used together with `edge_type`.
- `attribute_is_datetime` If specified together with `edge_attribute_label` it indicates that the type of attribute
to use for selecting edges to predict is a date with format dd/mm/yyyy. Currently, only date attributes are allowed so
this flag should always specified together with `edge_attribute_label`. Later implementations will support numeric
edge attributes in additions to dates.
- `edge_attribute_threshold <str>` For heterogeneous graphs and used together with `edge_attribute_label` it specifies
a value (date as only dates are currently supported) for edges to predict.
- `show_hist` If this flag is specified, then a histogram of the distances between source and target nodes comprising
negative edge samples is plotted. 

#### Examples

For the examples we use 2 different datasets. The **Cora** dataset that is a homogeneous network and the 
**BlogCatalog3** dataset that is a heterogeneous network. 

**Cora** can be downloaded from [here.](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)

**BlogCatalog3** can be downloaded from [here.]( http://socialcomputing.asu.edu/datasets/BlogCatalog3)

The **BlogCatalog3** dataset must be loaded into a `networkx` graph object. The `stellargraph` library provides a 
utility method, `stellargraph.data.loader.load_dataset_BlogCatalog3(location)`, that loads the dataset, 
and returns a `networkx` graph object. Assuming that the **BlogCatalog3** dataset has been downloaded and unzipped
in directory `~/data`, the following 3 lines of code will prepare the dataset for use in the below examples.

```
import os
from stellargraph.data.loader import load_dataset_BlogCatalog3
g = load_dataset_BlogCatalog3(location='~/data/BlogCatalog-dataset/data)
nx.write_gpickle(g, os.path.expanduser('~/data/BlogCatalog3.gpickle'))
```


**Example 1: Homogeneous graph with global sampling method for negative example** 
``` 
python main.py --input_graph=~/data/cora.cites --output_node_features=~/data/cora.emb --sampling_method='global'
```

**Example 2: Homogeneous graph with local sampling method for negative examples** 
``` 
python main.py --input_graph=~/data/cora.cites --output_node_features=~/data/cora.emb --sampling_method='local' --sampling_probs="0.0, 0.5, 0.5" --show_hist
```

**Example 3: Heterogeneous graph treated as homogeneous** 
``` 
python main.py --input_graph=~/data/BlogCatalog3.gpickle --output_node_features=~/data/bc3.emb --sampling_method='global'
```

**Example 4: Heterogeneous graph predicting edges based on edge type** 
``` 
python main.py --hin --input_graph=~/data/BlogCatalog3.gpickle --output_node_features=~/data/bc3.emb  --edge_type="friend" --sampling_method='global'
```

### References

1. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016. 

2. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017 

3. Social Computing Data Repository at ASU [http://socialcomputing.asu.edu]. R. Zafarani and H. Liu, (2009). Tempe, AZ: Arizona State University, School of Computing, Informatics and Decision Systems Engineering.