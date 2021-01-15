#!/usr/bin/env python3

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
import argparse
import copy
import difflib
import nbformat
import re
import shlex
import subprocess
import sys
import os
import tempfile
from itertools import chain
from traitlets import Set, Integer, Bool
from traitlets.config import Config
from pathlib import Path
from nbconvert import NotebookExporter, HTMLExporter, writers, preprocessors
from black import format_str, FileMode, InvalidInput

# determine the current stellargraph version
version = {}
with open("stellargraph/version.py", "r") as fh:
    exec(fh.read(), version)
SG_VERSION = version["__version__"]


PATH_RESOURCE_NAME = "notebook_path"


class ClearWarningsPreprocessor(preprocessors.Preprocessor):
    filter_all_stderr = Bool(True, help="Remove all stderr outputs.").tag(config=True)

    def preprocess(self, nb, resources):
        self.sub_warn = re.compile(r"^WARNING:tensorflow.*\n.*\n.*\n", re.MULTILINE)
        return super().preprocess(nb, resources)

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "code":
            pro_outputs = []
            for output in cell.outputs:
                # Search for tensorflow warning and remove warnings in outputs
                if "WARNING:tensorflow" in output.get("text", ""):
                    print(
                        f"Removing TensorFlow warning in code cell {cell.execution_count}"
                    )
                    output["text"] = self.sub_warn.sub("", output.get("text", ""))

                # Clear std errors
                if self.filter_all_stderr and output.get("name") == "stderr":
                    print(f"Removing stderr in code cell {cell.execution_count}")
                    continue

                pro_outputs.append(output)
            cell.outputs = pro_outputs
        return cell, resources


class RenumberCodeCellPreprocessor(preprocessors.Preprocessor):
    def preprocess(self, nb, resources):
        self.code_index = 0
        return super().preprocess(nb, resources)

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "code":
            self.code_index += 1
            cell.execution_count = self.code_index
        return cell, resources


class SetKernelSpecPreprocessor(preprocessors.Preprocessor):
    def preprocess(self, nb, resources):
        # Set the default kernel:
        if (
            "kernelspec" in nb.metadata
            and nb.metadata["kernelspec"]["name"] != "python3"
        ):
            print("Incorrect kernelspec:", nb.metadata["kernelspec"])

        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        return nb, resources


class FormatCodeCellPreprocessor(preprocessors.Preprocessor):
    linelength = Integer(90, help="Black line length.").tag(config=True)

    def preprocess(self, nb, resources):
        self.notebook_cells_changed = 0
        nb, resources = super().preprocess(nb, resources)
        if self.notebook_cells_changed > 0:
            print(f"Black formatted {self.notebook_cells_changed} code cells.")
        return nb, resources

    def preprocess_cell(self, cell, resources, cell_index):
        mode = FileMode(line_length=self.linelength)

        if cell.cell_type == "code":
            try:
                formatted = format_str(src_contents=cell["source"], mode=mode)
            except InvalidInput as err:
                print(f"Formatter error: {err}")
                formatted = cell["source"]

            if formatted and formatted[-1] == "\n":
                formatted = formatted[:-1]

            if cell["source"] != formatted:
                self.notebook_cells_changed += 1

            cell["source"] = formatted
        return cell, resources


def hide_cell_from_docs(cell):
    """
    Add metadata so that the cell is removed from the Sphinx output.

    https://nbsphinx.readthedocs.io/en/0.6.1/hidden-cells.html
    """
    cell["metadata"]["nbsphinx"] = "hidden"


class InsertTaggedCellsPreprocessor(preprocessors.Preprocessor):
    # abstract class working with tagged notebook cells
    metadata_tag = ""  # tag for added cells so that we can find them easily; needs to be set in derived class

    @staticmethod
    def tags(cell):
        return cell["metadata"].get("tags", [])

    @classmethod
    def remove_tagged_cells_from_notebook(cls, nb):
        # remove any tagged cells we added in a previous run
        nb.cells = [cell for cell in nb.cells if cls.metadata_tag not in cls.tags(cell)]

    @classmethod
    def tag_cell(cls, cell):
        cell["metadata"]["tags"] = [cls.metadata_tag]


class CloudRunnerPreprocessor(InsertTaggedCellsPreprocessor):
    metadata_tag = "CloudRunner"
    git_branch = "master"
    demos_path_prefix = "demos/"

    colab_import_code = f"""\
# install StellarGraph if running on Google Colab
import sys
if 'google.colab' in sys.modules:
  %pip install -q stellargraph[demos]=={SG_VERSION}"""

    def _binder_url(self, notebook_path):
        return f"https://mybinder.org/v2/gh/stellargraph/stellargraph/{self.git_branch}?urlpath=lab/tree/{notebook_path}"

    def _colab_url(self, notebook_path):
        return f"https://colab.research.google.com/github/stellargraph/stellargraph/blob/{self.git_branch}/{notebook_path}"

    def _binder_badge(self, notebook_path):
        # html needed to add the target="_parent" so that the links work from GitHub rendered notebooks
        return f'<a href="{self._binder_url(notebook_path)}" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a>'

    def _colab_badge(self, notebook_path):
        return f'<a href="{self._colab_url(notebook_path)}" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>'

    def _badge_markdown(self, notebook_path):
        # due to limited HTML-in-markdown support in Jupyter, place badges in an html table (paragraph doesn't work)
        return f"<table><tr><td>Run the latest release of this notebook:</td><td>{self._binder_badge(notebook_path)}</td><td>{self._colab_badge(notebook_path)}</td></tr></table>"

    def preprocess(self, nb, resources):
        notebook_path = resources[PATH_RESOURCE_NAME]
        if not notebook_path.startswith(self.demos_path_prefix):
            print(
                f"WARNING: Notebook file path of {notebook_path} didn't start with {self.demos_path_prefix}, and may result in bad links to cloud runners."
            )
        self.remove_tagged_cells_from_notebook(nb)
        badge_cell = nbformat.v4.new_markdown_cell(self._badge_markdown(notebook_path))
        self.tag_cell(badge_cell)
        # badges are created separately in docs by nbsphinx prolog / epilog
        hide_cell_from_docs(badge_cell)
        # the badges go after the first cell, unless the first cell is code
        if nb.cells[0].cell_type == "code":
            nb.cells.insert(0, badge_cell)
        else:
            nb.cells.insert(1, badge_cell)
        # find first code cell and insert a Colab import statement before it
        first_code_cell_id = next(
            index for index, cell in enumerate(nb.cells) if cell.cell_type == "code"
        )
        import_cell = nbformat.v4.new_code_cell(self.colab_import_code)
        self.tag_cell(import_cell)
        hide_cell_from_docs(import_cell)
        nb.cells.insert(first_code_cell_id, import_cell)

        nb.cells.append(badge_cell)  # add a badge to the bottom of notebook
        return nb, resources


class VersionValidationPreprocessor(InsertTaggedCellsPreprocessor):
    metadata_tag = "VersionCheck"

    version_check_code = f"""\
# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("{SG_VERSION}")
except AttributeError:
    raise ValueError(f"This notebook requires StellarGraph version {SG_VERSION}, but a different version {{sg.__version__}} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>.") from None"""

    def preprocess(self, nb, resources):
        self.remove_tagged_cells_from_notebook(nb)
        # find first (non-CloudRunner) code cell and insert before it
        first_code_cell_id = next(
            index
            for index, cell in enumerate(nb.cells)
            if cell.cell_type == "code"
            and CloudRunnerPreprocessor.metadata_tag not in self.tags(cell)
        )
        version_cell = nbformat.v4.new_code_cell(self.version_check_code)
        self.tag_cell(version_cell)
        hide_cell_from_docs(version_cell)
        nb.cells.insert(first_code_cell_id, version_cell)
        return nb, resources


class LoadingLinksPreprocessor(InsertTaggedCellsPreprocessor):
    metadata_tag = "DataLoadingLinks"
    search_tag = "DataLoading"

    data_loading_description = """\
(See [the "Loading from Pandas" demo]({}) for details on how data can be loaded.)"""

    def _relative_path(self, path):
        """
        Find the relative path from this notebook to the Loading Pandas one.

        This assumes that "demos" appears in the path, and is the root of the demos directories.
        """
        directories = os.path.dirname(path).split("/")
        demos_idx = next(
            index for index, directory in enumerate(directories) if directory == "demos"
        )
        nested_depth = len(directories) - (demos_idx + 1)
        parents = "../" * nested_depth
        return f"{parents}basics/loading-pandas.ipynb"

    def preprocess(self, nb, resources):
        self.remove_tagged_cells_from_notebook(nb)
        first_data_loading = next(
            (
                index
                for index, cell in enumerate(nb.cells)
                if self.search_tag in self.tags(cell)
            ),
            None,
        )

        if first_data_loading is not None:
            path = self._relative_path(resources[PATH_RESOURCE_NAME])
            links_cell = nbformat.v4.new_markdown_cell(
                self.data_loading_description.format(path)
            )
            self.tag_cell(links_cell)
            nb.cells.insert(first_data_loading, links_cell)

        return nb, resources


class IdempotentIdPreprocessor(preprocessors.Preprocessor):
    # https://github.com/jupyter/enhancement-proposals/blob/master/62-cell-id/cell-id.md introduces
    # 'cell ids', which nbformat 5.1.0+ inserts. However, it inserts random ones. This class
    # overwrites the random ones with fixed IDs.

    def preprocess_cell(self, cell, resources, cell_index):
        cell = copy.deepcopy(cell)
        cell.id = str(cell_index)
        return cell, resources


# ANSI terminal escape sequences
YELLOW_BOLD = "\033[1;33;40m"
LIGHT_RED_BOLD = "\033[1;91;40m"
RESET = "\033[0m"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Format and clean Jupyter notebooks by removing TensorFlow warnings "
        "and stderr outputs, formatting and numbering the code cells, and setting the kernel. "
        "See the options below to select which of these operations is performed."
    )
    parser.add_argument(
        "locations",
        nargs="+",
        help="Paths(s) to search for Jupyter notebooks to format",
    )
    parser.add_argument(
        "-w",
        "--clear_warnings",
        action="store_true",
        help="Clear TensorFlow  warnings and stderr in output",
    )
    parser.add_argument(
        "-c",
        "--format_code",
        action="store_true",
        help="Format all code cells (currently uses black)",
    )
    parser.add_argument(
        "-e",
        "--execute",
        nargs="?",
        const="default",
        help="Execute notebook before export with specified kernel (default if not given)",
    )
    parser.add_argument(
        "-t",
        "--cell_timeout",
        default=-1,
        type=int,
        help="Set the execution cell timeout in seconds (default is timeout disabled)",
    )
    parser.add_argument(
        "-n",
        "--renumber",
        action="store_true",
        help="Renumber all code cells from the top, regardless of execution order",
    )
    parser.add_argument(
        "-k",
        "--set_kernel",
        action="store_true",
        help="Set kernel spec to default 'Python 3'",
    )
    parser.add_argument(
        "-s",
        "--coalesce_streams",
        action="store_true",
        help="Coalesce streamed output into a single chunk of output",
    )
    parser.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="Perform default formatting, equivalent to -wcnksrv",
    )
    parser.add_argument(
        "-r",
        "--run_cloud",
        action="store_true",
        help="Add or update cells that support running this notebook via cloud services",
    )
    parser.add_argument(
        "-v",
        "--version_validation",
        action="store_true",
        help="Add or update cells that validate the version of StellarGraph",
    )
    parser.add_argument(
        "-l",
        "--loading_links",
        action="store_true",
        help="Add or update cells that link to docs for loading data",
    )
    parser.add_argument(
        "-i", "--ids", action="store_true", help="Add fixed IDs to each cell",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite original notebooks, otherwise a copy will be made with a .mod suffix",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Check that no changes happened, instead of writing the file",
    )
    group.add_argument(
        "--ci",
        action="store_true",
        help="Same as `--check`, but with an annotation for buildkite CI",
    )
    parser.add_argument(
        "--html", action="store_true", help="Save HTML as well as notebook output"
    )

    args, cmdline_args = parser.parse_known_args()

    # Ignore any notebooks in .ipynb_checkpoint directories
    ignore_checkpoints = True

    # Set other config from cmd args
    write_notebook = True
    write_html = args.html
    overwrite_notebook = args.overwrite
    check_notebook = args.check or args.ci
    on_ci = args.ci
    format_code = args.format_code or args.default
    clear_warnings = args.clear_warnings or args.default
    coalesce_streams = args.coalesce_streams or args.default
    renumber_code = args.renumber or args.default
    set_kernel = args.set_kernel or args.default
    execute_code = args.execute
    cell_timeout = args.cell_timeout
    run_cloud = args.run_cloud or args.default
    version_validation = args.version_validation or args.default
    loading_links = args.loading_links or args.default
    ids = args.ids or args.default

    # Add preprocessors
    preprocessor_list = []
    if run_cloud:
        preprocessor_list.append(CloudRunnerPreprocessor)
    if version_validation:
        preprocessor_list.append(VersionValidationPreprocessor)
    if loading_links:
        preprocessor_list.append(LoadingLinksPreprocessor)
    if renumber_code:
        preprocessor_list.append(RenumberCodeCellPreprocessor)

    if set_kernel:
        preprocessor_list.append(SetKernelSpecPreprocessor)

    if format_code:
        preprocessor_list.append(FormatCodeCellPreprocessor)

    if ids:
        # this needs to know the order of cells, so must run after all additions/changes
        preprocessor_list.append(IdempotentIdPreprocessor)

    if execute_code:
        preprocessor_list.append(preprocessors.ExecutePreprocessor)

    # these clean up the result of execution and so should happen after it
    if clear_warnings:
        preprocessor_list.append(ClearWarningsPreprocessor)

    if coalesce_streams:
        preprocessor_list.append(preprocessors.coalesce_streams)

    # Create the exporters with preprocessing
    c = Config()
    c.NotebookExporter.preprocessors = preprocessor_list
    c.HTMLExporter.preprocessors = preprocessor_list

    if execute_code:
        c.ExecutePreprocessor.timeout = cell_timeout
        if execute_code != "default":
            c.ExecutePreprocessor.kernel_name = execute_code

    nb_exporter = NotebookExporter(c)
    html_exporter = HTMLExporter(c)
    # html_exporter.template_file = 'basic'

    # Find all Jupyter notebook files in the specified directory
    all_files = []
    for p in args.locations:
        path = Path(p)
        if path.is_dir():
            all_files.extend(path.glob("**/*.ipynb"))
        elif path.is_file():
            all_files.append(path)
        else:
            raise ValueError(f"Specified location not '{path}'a file or directory.")

    check_failed = []

    # Go through all notebooks files in specified directory
    for file_loc in all_files:
        # Skip Modified files
        if "mod" in str(file_loc):
            continue
        # Skip checkpoints
        if ignore_checkpoints and ".ipynb_checkpoint" in str(file_loc):
            continue

        print(f"{YELLOW_BOLD} \nProcessing file {file_loc}{RESET}")
        in_notebook = nbformat.read(str(file_loc), as_version=4)

        # the CloudRunnerPreprocessor needs to know the filename of this notebook
        resources = {PATH_RESOURCE_NAME: str(file_loc)}

        writer = writers.FilesWriter()

        if write_notebook:
            # Process the notebook to a new notebook
            (body, resources) = nb_exporter.from_notebook_node(
                in_notebook, resources=resources
            )

            temporary_file = None

            # Write notebook file
            if overwrite_notebook:
                nb_file_loc = str(file_loc.with_suffix(""))
            elif check_notebook:
                tempdir = tempfile.TemporaryDirectory()
                nb_file_loc = f"{tempdir.name}/notebook"
            else:
                nb_file_loc = str(file_loc.with_suffix(".mod"))

            print(f"Writing notebook to {nb_file_loc}.ipynb")
            writer.write(body, resources, nb_file_loc)

            if check_notebook:
                with open(file_loc) as f:
                    original = f.read()

                with open(f"{nb_file_loc}.ipynb") as f:
                    updated = f.read()

                if original != updated:
                    check_failed.append(str(file_loc))

                    if on_ci:
                        # CI doesn't provide enough state to diagnose a peculiar or
                        # seemingly-spurious difference, so include a diff in the logs. This allows
                        # us to inspect the change retroactive if required, but doesn't junk up the
                        # final output/annotation.
                        sys.stdout.writelines(
                            difflib.unified_diff(
                                original.splitlines(keepends=True),
                                updated.splitlines(keepends=True),
                            )
                        )

                        if "GITHUB_ACTIONS" in os.environ:
                            # special annotations for github actions
                            print(
                                f"::error file={file_loc}::Notebook failed format check. Fix by running:%0A"
                                f"python ./scripts/format_notebooks.py --default --overwrite {file_loc}"
                            )

                tempdir.cleanup()

        if write_html:
            # Process the notebook to HTML
            (body, resources) = html_exporter.from_notebook_node(
                in_notebook, resources=resources
            )

            html_file_loc = str(file_loc.with_suffix(""))
            print(f"Writing HTML to {html_file_loc}.html")
            writer.write(body, resources, html_file_loc)

    if check_failed:
        assert check_notebook, "things failed check without check being enabled"

        notebooks = "\n".join(f"- `{path}`" for path in check_failed)

        command = "python ./scripts/format_notebooks.py --default --overwrite demos/"

        message = f"""\
Found notebook(s) with incorrect formatting:

{notebooks}

Fix by running:

    {command}"""

        print(f"\n{LIGHT_RED_BOLD}Error:{RESET} {message}")

        if on_ci:
            try:
                subprocess.run(
                    [
                        "buildkite-agent",
                        "annotate",
                        "--style=error",
                        "--context=format_notebooks",
                        message,
                    ]
                )
            except FileNotFoundError:
                # no agent, so probably not on buildkite, and so silently continue without an annotation
                pass

        sys.exit(1)
