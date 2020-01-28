## Development Scripts Readme

This directory contains scripts to perform testing and maintainence tasks for developers of the StellarGraph  library. These scripts should not be needed for users of the library.

Currently there are two scripts, one to formats and clean the specified Juptyer notebooks, the other to run specific demo notebooks and check if there are any errors.  In the future the functionality of these two scripts may be combined.

### Requirements

These scripts have requirements in addition to the base library, install the requirements using the `test` extra, e.g. in the parent directory:

```
pip install .[test]
```


### Format and clean up demo Jupyter notebooks

The script `format_notebooks.py` uses `nbconvert` to process all code and output cells in the specified locations, optionally remove Tensorflow warnings and stderr outputs, format and number the code cells, and set the kernel to the default of "Python 3".

The usage options are available with the `--help` command line argument:
```
> format_notebooks.py --help
```

#### Example usage

To perform all formatting on all Jupyter notebooks found in the demos directory, execute the following command in the top-level stellargraph directory:
```
> python scripts/format_notebooks.py -d demos
```

To additionally output HTML files:
```
> python scripts/format_notebooks.py -d --html demos
```

To format and execute all Jupyter notebooks found in the demos directory, overwriting them with the updated versions:
```
> python scripts/format_notebooks.py -e -o -d demos
```

### Testing demo Jupyter notebooks

The `test_demos.py` script runs the demo scripts and Jupyter notebooks that use the CORA dataset, it currently skips the demos that use other datasets (due to dataset licencing issues) and those that require complicated dependencies (such as iGraph which requires a pre-compiled library not installable through PyPI).  The script has a dependency on `treon` which is used to test the notebooks.

The script currently has no command line arguments and is just run as follows:
```
python scripts/test_demos.py
```

The script takes a while to run, as it loops through all supported demo notebooks and scripts. FInally, it will print the number of passed and failed notebooks and scripts.
