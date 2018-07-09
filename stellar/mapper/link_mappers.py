# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mappers to provide input data for link prediction/link attribute inference problems using GraphSAGE and HinSAGE.

"""

import networkx as nx
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Tuple, Callable, Optional
from keras.utils import Sequence

from stellar.data.explorer import SampledBreadthFirstWalk

class GraphSAGELinkMapper(Sequence):
    """Keras-compatible link data mapper for link prediction using Homogeneous GraphSAGE

    Args:
        ids: Link IDs to batch, each link id being a tuple of (src, dst) node ids.
            (The graph nodes must have a "feature" attribute that is used as input to the GraphSAGE model.)
            These are the links that are to be used to train or inference, and the embeddings
            calculated for these links via a binary operator applied to their source and destination nodes,
            are passed to the downstream task of link prediction or link attribute inference.
            The source and target nodes of the links are used as head nodes for which subgraphs are sampled.
            The subgraphs are sampled from all nodes.
        link_labels: Labels of the above links, e.g., 0 or 1 for the link prediction problem.
        sampler: A node sampler instance defined on graph G.
        batch_size: Size of batch of links to return.
        num_samples: List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        feature_size: Node feature size (optional)
        name: Name of mapper
    """

    def __init__(
            self,
            ids: List[Tuple[Any, Any]], # allow for node ids to be anything, e.g., str or int
            link_labels: List[Any] or np.ndarray,
            sampler: Callable[[List[Any]], List[List[Any]]],
            batch_size: int,
            num_samples: List[int],
            feature_size: Optional[int] = None,
            name: AnyStr = None,
    ):
        self.G = sampler.graph
        self.sampler = sampler
        self.num_samples = num_samples
        self.labels = link_labels

        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name

        # Check correct graph sampler is used
        if not isinstance(sampler, SampledBreadthFirstWalk):
            raise TypeError(
                "Sampler must be an instance of SampledBreadthFirstWalk class"
            )

        # Ensure features are available:
        nodes_have_features = all(
            ["feature" in vdata for v, vdata in self.G.nodes(data=True)]
        )
        if not nodes_have_features:
            print("Warning: Not all nodes have a 'feature' attribute.")
            print("Warning: This is required for the GraphSAGE mapper.")

        # Check that all nodes have features of the same size
        # Note: if there are no features in the nodes this will be 1!
        feature_sizes = {
            np.size(vdata.get("feature")) for v, vdata in self.G.nodes(data=True)
        }

        if feature_size:
            self.feature_size = feature_size
        else:
            self.feature_size = int(max(feature_sizes))
            if len(feature_sizes) > 1:
                print("Warning: feature sizes in nodes inconsistent (using max)")
                print("Found feature sizes: {}".format(feature_sizes))

        if self.feature_size not in feature_sizes:
            print("Found feature sizes: {}".format(feature_sizes))
            raise RuntimeWarning("Specified feature size doesn't match graph features")





