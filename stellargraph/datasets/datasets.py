# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample datasets for stellargraph demonstrations
"""

from .dataset_loader import DatasetLoader


class Cora(DatasetLoader):
    def __init__(self) -> None:
        super(Cora, self).__init__(
            "Cora",
            "cora",
            "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
            "gztar",
            expected_files=["cora.cites", "cora.content"],
            description="The Cora dataset consists of 2708 scientific publications classified into one of seven classes. "
            "The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector "
            "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.",
            source="https://linqs.soe.ucsc.edu/data",
        )


class CiteSeer(DatasetLoader):
    def __init__(self) -> None:
        super(CiteSeer, self).__init__(
            "CiteSeer",
            "CiteSeer",
            "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
            "gztar",
            expected_files=["citeseer.cites", "citeseer.content"],
            description="The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. "
            "The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector "
            "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.",
            source="https://linqs.soe.ucsc.edu/data",
        )


class PubMedDiabetes(DatasetLoader):
    """
    The Pubmed Diabetes dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes.
    The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.

    Further details at: https://linqs.soe.ucsc.edu/data
    """

    def __init__(self) -> None:
        super(PubMedDiabetes, self).__init__(
            "PubMed Diabetes",
            "Pubmed-Diabetes",
            "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz",
            "gztar",
            expected_files=[
                "data/Pubmed-Diabetes.DIRECTED.cites.tab",
                "data/Pubmed-Diabetes.GRAPH.pubmed.tab",
                "data/Pubmed-Diabetes.NODE.paper.tab",
            ],
            description="The PubMed Diabetes dataset consists of 19717 scientific publications from PubMed database "
            "pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. "
            "Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.",
            source="https://linqs.soe.ucsc.edu/data",
            data_subdirectory_name="data",
        )


class BlogCatalog3(DatasetLoader):
    def __init__(self) -> None:
        super(BlogCatalog3, self).__init__(
            "BlogCatalog3",
            "BlogCatalog-dataset",
            "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip",
            "zip",
            expected_files=[
                "data/edges.csv",
                "data/group-edges.csv",
                "data/groups.csv",
                "data/nodes.csv",
            ],
            description="This dataset is crawled from a social blog directory website BlogCatalog "
            "http://www.blogcatalog.com and contains the friendship network crawled and group memberships.",
            source="http://socialcomputing.asu.edu/datasets/BlogCatalog3",
            data_subdirectory_name="data",
        )


class MovieLens(DatasetLoader):
    def __init__(self) -> None:
        super(BlogCatalog3, self).__init__(
            "MovieLens",
            "ml-100k",
            "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "zip",
            expected_files=["u.data", "u.user", "u.item", "u.genre", "u.occupation",],
            description="MovieLens 100K movie ratings. Stable benchmark dataset. "
            "100,000 ratings from 1000 users on 1700 movies. Released 4/1998.",
            source="https://grouplens.org/datasets/movielens/100k/",
        )
