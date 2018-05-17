# Link Prediction demo

The main.py script runs link prediction on a homogeneous or heterogeneous graph. When
the graph is heterogeneous, it is treated as homogeneous for representation learning; in
this case, the link prediction script can be thought of us a baseline for more advanced
algorithms that do not simplify the input graph.

### Instalation
To install the requirements for running the script, create a new python v3.6 environment
and execute,

pip install -r requirements.txt

### Usage

#### Input Data
The input data should be in EPGM format and the directory specified on the command line.

#### Command Line Arguments
The script has two command line arguments as follows:

- `input_graph <str: directory>`  The directory where the graph in EPGM format is stored.
- `output_node_features <str: filename>` The file where the node features from representation learning are written 
for future reference.
- `sampling_method <str: 'local' or global'>` Select how pairs of nodes that are not connected are selected as negative 
edge samples. The 'global' method, selects pairs of nodes uniformly at random from all nodes in the graph that are not
connected by an edge. The 'local' methods first samples a distance (number of edges) from a source node (selected
uniformly at random from all nodes int he graph) and then selects the first node at this distance as the target node
for the negative edge sample. The distance is sampled based on a probability distribution that can be specified using
the probs command line parameter.
- `sampling_probs <str: comma separated list of floats>` The probability distribution for sampling distances between 
source and target nodes. The first value should always be 0.0 and the remaining values should sum to one. An example for 
sampling target nodes up to 4 edges away from source is `"0.0, 0.25, 0.25, 0.5"`.
- `show_hist` If this flag is specified, then a histogram of the distances between source and target nodes comprising
negative edge samples is plotted. 


**Example 1:** 
``` 
python main.py --input_graph=~/data/cora.epgm/ --output_node_features=~/data/cora.emb --sampling_method='global'
```

**Example 2:** 
``` 
python main.py --input_graph=~/data/cora.epgm/ --output_node_features=~/data/cora.emb --sampling_method='local' --sampling_probs="0.0, 0.5, 0.5" --show_hist
