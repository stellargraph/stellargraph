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
import nbformat
import re
import shlex
import subprocess
import sys
import tempfile
from itertools import chain
from traitlets import Set, Integer, Bool
from traitlets.config import Config
from pathlib import Path
from nbconvert import NotebookExporter, HTMLExporter, writers, preprocessors

from black import format_str, FileMode, InvalidInput


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
                        f"Removing Tensorflow warning in code cell {cell.execution_count}"
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


class CloudRunnerPreprocessor(preprocessors.Preprocessor):
    path_resource_name = "cloud_runner_path"
    metadata_tag = "CloudRunner"  # tag for added cells so that we can find them easily
    git_branch = "master"

    colab_import_code = """\
# install StellarGraph if running on Google Colab
import sys
if 'google.colab' in sys.modules:
  !pip install -q stellargraph[demos]"""

    def _binder_url(self, notebook_path):
        return f"https://mybinder.org/v2/gh/stellargraph/stellargraph/{self.git_branch}?filepath={notebook_path}"

    def _colab_url(self, notebook_path):
        return f"https://colab.research.google.com/github/stellargraph/stellargraph/blob/{self.git_branch}/{notebook_path}"

    def _binder_badge(self, notebook_path):
        return f'<a href="{self._binder_url(notebook_path)}" alt="Open In Binder"><img src="https://mybinder.org/badge_logo.svg"/>'

    def _colab_badge(self, notebook_path):
        return f'<a href="{self._colab_url(notebook_path)}" alt="Open In Colab"><img src="https://colab.research.google.com/assets/colab-badge.svg"/>'

    def preprocess(self, nb, resources):
        notebook_path = resources[self.path_resource_name]
        if not notebook_path.startswith("demos/"):
            print(f"Notebook file path of {notebook_path} didn't start with demo")
        # remove any cells we added in a previous run
        nb.cells = [
            cell
            for cell in nb.cells
            if self.metadata_tag not in cell["metadata"].get("tags", [])
        ]
        # due to limited HTML-in-markdown support in Jupyter, place badges in an html table (paragraph doesn't work)
        badge_markdown = f"<table><tr><td>{self._binder_badge(notebook_path)}</td><td>{self._colab_badge(notebook_path)}</td></tr></table>"
        badge_cell = nbformat.v4.new_markdown_cell(badge_markdown)
        badge_cell["metadata"]["tags"] = [self.metadata_tag]
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
        import_cell["metadata"]["tags"] = [self.metadata_tag]
        nb.cells.insert(first_code_cell_id, import_cell)

        nb.cells.append(badge_cell)  # add a badge to the bottom of notebook
        return nb, resources


# ANSI terminal escape sequences
YELLOW_BOLD = "\033[1;33;40m"
LIGHT_RED_BOLD = "\033[1;91;40m"
RESET = "\033[0m"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Format and clean Jupyter notebooks by removing Tensorflow warnings "
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
        help="Clear Tensorflow  warnings and stderr in output",
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
        help="Perform default formatting, equivalent to -wcnksr",
    )
    parser.add_argument(
        "-r",
        "--run_cloud",
        action="store_true",
        help="Add or update cells that support running this notebook via cloud services",
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

    # Add preprocessors
    preprocessor_list = []
    if run_cloud:
        preprocessor_list.append(CloudRunnerPreprocessor)
    if renumber_code:
        preprocessor_list.append(RenumberCodeCellPreprocessor)

    if set_kernel:
        preprocessor_list.append(SetKernelSpecPreprocessor)

    if format_code:
        preprocessor_list.append(FormatCodeCellPreprocessor)

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
        resources = {CloudRunnerPreprocessor.path_resource_name: str(file_loc)}

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
            subprocess.run(
                [
                    "buildkite-agent",
                    "annotate",
                    "--style=error",
                    "--context=format_notebooks",
                    message,
                ]
            )

        sys.exit(1)
