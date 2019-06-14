#!/usr/bin/env python
import os
import sys
import subprocess
import warnings

SGDIR = os.path.realpath(os.path.dirname(sys.argv[0]) + "/..")

def test_notebooks():
    # Jupyter notebooks to test:
    notebook_paths = ["demos/node-classification/gat/", "demos/node-classification/graphsage/", 
                      "demos/node-classification/node2vec/", "demos/node-classification/sgc/", "demos/link-prediction/graphsage/", "demos/link-prediction/hinsage/", 
                      "demos/link-prediction/random-walks/", "demos/calibration", "demos/embeddings","demos/ensembles"]

    num_errors = 0
    for nb_path in notebook_paths:
        cmd_line = f"treon . --threads=1"

        print(f"\033[1;33;40m Running {nb_path}\033[0m")

        # Add path to PYTHONPATH
        environ = dict(os.environ, PYTHONPATH=nb_path)

        procout = subprocess.run(
            cmd_line,
            shell=True,
            check=False,
            env=environ,
            cwd=nb_path,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )

        if procout.returncode != 0:
            num_errors += 1
            print("{}  \033[1;31;40m -- FAILED\033[0m".format(nb_path))

        else:
            print("{}  \033[1;32;40m -- SUCCEEDED\033[0m".format(nb_path))

        print()
    return num_errors

def test_scripts():
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
       
    ]

    # Datasets & arguments
    argument_dict = {"CORA_DIR": os.path.expandvars("$HOME/data/cora")}
    num_errors = 0
    for script in scripts:
        cmd_args = script["args"].format(**argument_dict)
        cmd_line = "python " + script["path"] + " " + cmd_args

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
            print("{path}  \033[1;31;40m -- FAILED\033[0m".format(**script))

        else:
            print("{path}  \033[1;32;40m -- SUCCEEDED\033[0m".format(**script))

        print()
    return num_errors

if __name__=="__main__":
    num_errors = test_notebooks()
    num_errors += test_scripts()

    if num_errors > 0:
        exit(1)
    else:
        exit(0)