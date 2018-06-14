# Link Prediction demo

The main.py script runs link prediction on a homogeneous or heterogeneous graph. When
the graph is heterogeneous, it is treated as homogeneous for representation learning; in
this case, the link prediction script can be thought of us a baseline for more advanced
algorithms that do not simplify the input graph. 

In addition, for heterogeneous graphs, the link prediction script gives the user some control over what edge 
types to predict including a choice of filtering these edges by one of their attributes. Currently, the script only
allows filtering by a date attribute in the format **dd/mm/yyyy**, e.g., *10/10/2005*. The edge attribute holding the date
can be given any text label in the graph as stored on disk in EPGM format, e.g., 'date', 'timestamp', 'start date', etc. 

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

#### Input Data
The input data should be in EPGM format and the directory specified on the command line.

#### Command Line Arguments
The script accepts the following command line arguments:
               
- `input_graph <str>`  The directory where the graph in EPGM format is stored.
- `output_node_features <str>` The file where the node features from representation learning are written 
for future reference.
- `subsample` If specified, the graph is subsampled (number of nodes reduced) by a default 0.1 factor, e.g,
10 percent of the original graph. This option is useful for speeding up testing on large graphs.
- `subgrap_size <float>` The size of the graph when the `subsample` option is given. It should be a value
in the intervale (0,1).
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


**Example 1: Homogeneous graph with global sampling method for negative example** 
``` 
python main.py --input_graph=~/data/cora.epgm/ --output_node_features=~/data/cora.emb --sampling_method='global'
```

**Example 2: Homogeneous graph with local sampling method for negative examples** 
``` 
python main.py --input_graph=~/data/cora.epgm/ --output_node_features=~/data/cora.emb --sampling_method='local' --sampling_probs="0.0, 0.5, 0.5" --show_hist
```

**Example 3: Heterogeneous graph treated as homogeneous** 
``` 
python main.py --input_graph=~/data/BlogCatalog3.epgm/ --dataset_name="Blog Catalog3" --output_node_features=~/data/bc3.emb --sampling_method='global'
```

**Example 4: Heterogeneous graph predicting edges based on edge type and property** 
``` 
python main.py --hin --input_graph=~/data/BlogCatalog3.epgm/ --dataset_name="Blog Catalog3" --output_node_features=~/data/bc3.emb  --edge_type="friend" --edge_attribute_label="date" --attribute_is_datetime --edge_attribute_threshold="01/01/2005" --sampling_method='global'
```