# -*- coding: utf-8 -*-
#
# Copyright 2019 Data61, CSIRO
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
#!/usr/bin/env python3
import argparse
import nbformat
import re
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
        default=None,
        const="default",
        help="Execute notebook before export with specified kernel (default if not given)",
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
        "-d",
        "--default",
        action="store_true",
        help="Perform default formatting, equivalent to -wcnk",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite original notebooks, otherwise a copy will be made with a .mod suffix",
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
    format_code = args.format_code or args.default
    clear_warnings = args.clear_warnings or args.default
    renumber_code = args.renumber or args.default
    set_kernel = args.set_kernel or args.default
    execute_code = args.execute

    # Add preprocessors
    preprocessor_list = []
    if renumber_code:
        preprocessor_list.append(RenumberCodeCellPreprocessor)

    if clear_warnings:
        preprocessor_list.append(ClearWarningsPreprocessor)

    if set_kernel:
        preprocessor_list.append(SetKernelSpecPreprocessor)

    if format_code:
        preprocessor_list.append(FormatCodeCellPreprocessor)

    if execute_code:
        preprocessor_list.append(preprocessors.ExecutePreprocessor)

    # Create the exporters with preprocessing
    c = Config()
    c.NotebookExporter.preprocessors = preprocessor_list
    c.HTMLExporter.preprocessors = preprocessor_list

    if execute_code and execute_code != "default":
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

    # Go through all notebooks files in specified directory
    for file_loc in all_files:
        # Skip Modified files
        if "mod" in str(file_loc):
            continue
        # Skip checkpoints
        if ignore_checkpoints and ".ipynb_checkpoint" in str(file_loc):
            continue

        print(f"\033[1;33;40m \nProcessing file {file_loc}\033[0m")
        in_notebook = nbformat.read(str(file_loc), as_version=4)

        writer = writers.FilesWriter()

        if write_notebook:
            # Process the notebook to a new notebook
            (body, resources) = nb_exporter.from_notebook_node(in_notebook)

            # Write notebook file
            if overwrite_notebook:
                nb_file_loc = str(file_loc.with_suffix(""))
            else:
                nb_file_loc = str(file_loc.with_suffix(".mod"))
            print(f"Writing notebook to {nb_file_loc}.ipynb")
            writer.write(body, resources, nb_file_loc)

        if write_html:
            # Process the notebook to HTML
            (body, resources) = html_exporter.from_notebook_node(in_notebook)

            html_file_loc = str(file_loc.with_suffix(""))
            print(f"Writing HTML to {html_file_loc}.html")
            writer.write(body, resources, html_file_loc)
