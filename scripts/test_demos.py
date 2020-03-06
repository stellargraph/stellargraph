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
The StellarGraph class that encapsulates information required for
a machine-learning ready graph used by models.

"""
#!/usr/bin/env python
#
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

import os
import sys
import subprocess
import warnings

# The stellargraph directory
SGDIR = os.path.realpath(os.path.dirname(sys.argv[0]) + "/..")
print(SGDIR)

# Jupyter notebooks to test:
notebook_paths = [
    "demos/node-classification/attri2vec/",
    "demos/node-classification/gat/",
    "demos/node-classification/graphsage/",
    "demos/node-classification/node2vec/",
    "demos/node-classification/sgc/",
    # "demos/link-prediction/attri2vec/",
    "demos/link-prediction/graphsage/",
    # "demos/link-prediction/hinsage/",
    "demos/link-prediction/random-walks/",
    # "demos/interpretability/gcn/",
    "demos/calibration",
    "demos/embeddings",
    "demos/ensembles",
]


def test_notebooks():
    """
    Run all notebooks in the directories given by the list `notebook_paths`.

    The notebooks are run locally using [treon](https://github.com/ReviewNB/treon)
    and executed in each directory so that local resources can be imported.

    Returns:
        num_errors (int): Number of notebooks that failed to run
        num_passed (int): Number of notebooks that successfully run
    """
    num_errors = 0
    num_passed = 0
    for nb_path in notebook_paths:
        abs_nb_path = os.path.join(SGDIR, nb_path)
        cmd_line = f"treon . --threads=2"

        print(f"\033[1;33;40m Running {abs_nb_path}\033[0m")

        # Add path to PYTHONPATH
        environ = dict(os.environ, PYTHONPATH=abs_nb_path)

        procout = subprocess.run(
            cmd_line,
            shell=True,
            check=False,
            env=environ,
            cwd=abs_nb_path,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )

        if procout.returncode != 0:
            num_errors += 1

        else:
            num_passed += 1

        print()
    return num_errors, num_passed


if __name__ == "__main__":
    num_errors_nb, num_passed_nb = test_notebooks()

    print("=" * 100)
    print("\033[1;31;40m" if num_errors_nb > 0 else "\033[1;32;40m")
    print(f"Demo notebooks: {num_passed_nb} passed and {num_errors_nb} failed")
    print("\033[0m")
    print("=" * 100)

    if num_errors_nb > 0:
        exit(1)
    else:
        exit(0)
