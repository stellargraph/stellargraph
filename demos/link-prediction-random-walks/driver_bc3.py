import networkx as nx
import os
import pandas as pd
from stellargraph.data.loader import load_dataset_BlogCatalog3


# def load_dataset_BlogCatalog3(location):
#
#     location = os.path.expanduser(location)
#     if not os.path.isdir(location):
#         print("The location {} is not a directory.".format(location))
#         exit(0)
#
#     # load the raw data
#     user_node_ids = pd.read_csv(os.path.join(location, "nodes.csv"), header=None)
#     group_ids = pd.read_csv(os.path.join(location, "groups.csv"), header=None)
#     edges = pd.read_csv(os.path.join(location, "edges.csv"), header=None)
#     group_edges = pd.read_csv(os.path.join(location, "group-edges.csv"), header=None)
#
#     # convert the dataframes to lists because that is what networkx expects as input
#     user_node_ids = user_node_ids[0].tolist()
#     group_ids = group_ids[0].tolist()
#     edges = list(edges.itertuples(index=False, name=None))  # convert to list of tuples
#     group_edges = list(group_edges.itertuples(index=False, name=None))
#
#     # The dataset uses integers for node ids. However, the integers from 1 to 39 are used as IDs for both users and
#     # groups. This would cause a confusion when constructing the networkx graph object. As a result, we convert all
#     # IDs to string and append the character 'p' to the integer ID for user nodes and the character 'g' to the integer
#     # ID for group nodes.
#     user_node_ids = ['p'+str(user_node_id) for user_node_id in user_node_ids]
#     group_ids = ['g'+str(group_id) for group_id in group_ids]
#     edges = [('p'+str(from_node), 'p'+str(to_node)) for from_node, to_node in edges]
#     group_edges = [('p'+str(from_node), 'g'+str(to_node)) for from_node, to_node in group_edges]
#
#     g_nx = nx.Graph()  # create the graph
#
#     # add user and group nodes with labels 'Person' and 'Group' respectively.
#     g_nx.add_nodes_from(user_node_ids, label='Person')
#     g_nx.add_nodes_from(group_ids, label='Group')
#
#     # add the user-user edges with label 'friend'
#     g_nx.add_edges_from(edges, label='friend')
#
#     # add user-group edges with label 'belongs'
#     g_nx.add_edges_from(group_edges, label='belongs')
#
#     return g_nx


if __name__ == "__main__":

    location = "~/Projects/data/BlogCatalog3/BlogCatalog-dataset/data"
    g = load_dataset_BlogCatalog3(location=location)
    nx.write_gpickle(g, os.path.expanduser('~/data/BlogCatalog3.gpickle'))
