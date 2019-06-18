#!/usr/bin/env python

import os
import sys
import subprocess
import warnings

# The stellargraph directory
SGDIR = os.path.realpath(os.path.dirname(sys.argv[0]) + "/..")
print(SGDIR)

# Scripts to run:
scripts = [
    {
        "name": "GAT Demo script",
        "path": "demos/node-classification/gat/gat-cora-example.py",
        "args": "-l {CORA_DIR}",
    },
    {
        "name": "GCN Demo script",
        "path": "demos/node-classification/gcn/gcn-cora-example.py",
        "args": "-l {CORA_DIR}",
    },
    {
        "name": "GraphSAGE Demo script",
        "path": "demos/node-classification/graphsage/graphsage-cora-example.py",
        "args": "-l {CORA_DIR}",
    },
    {
        "name": "GraphSAGE LP Demo script",
        "path": "demos/link-prediction/graphsage/cora-links-example.py",
        "args": "-l {CORA_DIR}",
    },
    {
        "name": "HinSAGE Movielens Demo script",
        "path": "demos/link-prediction/hinsage/movielens-recommender.py",
        "args": "-p {ML100_DIR} -f {SCRIPT_DIR}/ml-100k-config.json",
    },
]

# Datasets & arguments
argument_dict = {
    "CORA_DIR": os.path.expandvars("$HOME/data/cora"),
    "ML100_DIR": os.path.expandvars("$HOME/data/ml-100k"),
}

# Jupyter notebooks to test:
notebook_paths = [
    "demos/node-classification/gat/",
    "demos/node-classification/graphsage/",
    "demos/node-classification/node2vec/",
    "demos/node-classification/sgc/",
    "demos/link-prediction/graphsage/",
    "demos/link-prediction/hinsage/",
    "demos/link-prediction/random-walks/",
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


def test_scripts():
    """
    Run all scripts in the dictionary `scripts`.

    Returns:
        num_errors (int): Number of scripts that failed to run
        num_passed (int): Number of scripts that successfully run
    """
    num_errors = 0
    num_passed = 0
    for script in scripts:
        abs_script_path = os.path.join(SGDIR, script["path"])
        argument_dict['SCRIPT_DIR'] =  os.path.dirname(abs_script_path)

        cmd_args = script["args"].format(**argument_dict)
        cmd_line = "python " + abs_script_path + " " + cmd_args

        print("\033[1;33;40m Running {path}\033[0m".format(**script))

        procout = subprocess.run(
            cmd_line,
            shell=True,
            check=False,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )
        script["success"] = procout.returncode == 0
        # script["output"] = procout.stdout.decode("ascii")

        if procout.returncode != 0:
            num_errors += 1
            print("> {path}  \033[1;31;40m -- FAILED\033[0m".format(**script))

        else:
            num_passed += 1
            print("> {path}  \033[1;32;40m -- SUCCEEDED\033[0m".format(**script))

        print()
    return num_errors, num_passed


if __name__ == "__main__":
    num_errors_nb, num_passed_nb = test_notebooks()
    num_errors_sc, num_passed_sc = test_scripts()

    print("=" * 100)
    print("\033[1;31;40m" if num_errors_sc > 0 else "\033[1;32;40m")
    print(f"Demo scripts: {num_passed_sc} passed and {num_errors_sc} failed")
    print("\033[1;31;40m" if num_errors_nb > 0 else "\033[1;32;40m")
    print(f"Demo notebooks: {num_passed_nb} passed and {num_passed_nb} failed")
    print("\033[0m")
    print("=" * 100)

    if num_errors_nb > 0 or num_errors_sc > 0:
        exit(1)
    else:
        exit(0)
