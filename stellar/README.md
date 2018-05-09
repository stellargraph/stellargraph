## Example Library Usage

**It's all made up and the code will certainly not run; the examples are meant to help guide the library design
process**

```
from stellar.representation_learning import *
from stellar.graph_explorer import *
from stellar.data_splitter import *

def link_prediction_clf(edge_embeddings, edge_data):
    # Function that trains a link prediction classifier. It could be any classifier from
    # scikit-learn, e.g., RandomForrest, LogisticRegression, etc., or a Neural Network classifier implemented
    # using Keras or Tensorflow.
    return clf

def predict_links(edge_embeddings, edge_data, clf): 
    # Predict on the edge_data given the classifier. Care must be taken to correclty format data for
    # classification; also it is assumed that a real program will take care to create datasets for training
    # and testing correctly.
    return clf.predict(edge_data)   

# Load graph in EPGM format
graph_dir = '/data/dataset.epgm/'
graph = EPGM(graph_dir) 
graph = graph.to_nx()  # convert to networkx format


# Node splitter object for node attribute inference
node_splitter = NodeSplitter(graph=graph, graph_master=None)

# Edge splitter object for link prediction
edge_splitter = EdgeSplitter(graph=graph, graph_master=None)

g_train, edge_data = edge_splitter.train_test_split(p=0.25, method='global')

# Representation learning using random walks
edge_rl_obj = EdgeRepresentationLearning(g_train)
rw_explorer_obj = BiasedRandomWalk(p=0.5, q = 2, n=10, l=100, e_types=None)  # homogeneous graph
edge_rl_obj.fit(method='biased-random-walk', random_walk_explorer=rw_explorer_obj, binary_operator='l2')

# Retrieve the edge embedding as typles (source node, targer node, embedding vector)
edge_embeddings = edge_rl_obj.embeddings  

# Train the link prediction classifier using edge_embeddings and edge_data
clf = link_prediction_clf(edge_embeddings, edge_data)
y_pred = predict_links(edge_embeddings, edge_data, clf)

# now do model evaluation etc.

```