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

    parser = argparse.ArgumentParser(description="Format Jupyter notebooks")
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

    # Default to always writing the notebook
    write_notebook = True

    # Add preprocessors
    preprocessor_list = []
    if args.renumber:
        preprocessor_list.append(RenumberCodeCellPreprocessor)

    if args.clear_warnings:
        preprocessor_list.append(ClearWarningsPreprocessor)

    if args.set_kernel:
        preprocessor_list.append(SetKernelSpecPreprocessor)

    if args.format_code:
        preprocessor_list.append(FormatCodeCellPreprocessor)

    # Create the exporters with preprocessing
    c = Config()
    c.NotebookExporter.preprocessors = preprocessor_list
    c.HTMLExporter.preprocessors = preprocessor_list
    nb_exporter = NotebookExporter(c)
    html_exporter = HTMLExporter(c)
    # html_exporter.template_file = 'basic'

    # Find all Jupyter notebook files in the specified directory
    all_files = chain.from_iterable(Path(p).glob("**/*.ipynb") for p in args.locations)

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
            if args.overwrite:
                nb_file_loc = str(file_loc.with_suffix(""))
            else:
                nb_file_loc = str(file_loc.with_suffix(".mod"))
            print(f"Writing notebook to {nb_file_loc}.ipynb")
            writer.write(body, resources, nb_file_loc)

        if args.html:
            # Process the notebook to HTML
            (body, resources) = html_exporter.from_notebook_node(in_notebook)

            html_file_loc = str(file_loc.with_suffix(""))
            print(f"Writing HTML to {html_file_loc}.html")
            writer.write(body, resources, html_file_loc)

