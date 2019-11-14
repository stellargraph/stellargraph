#!/usr/bin/env python3

import nbformat
import re

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
            # Search and remove warnings in outputs
            pro_outputs = []
            for output in cell.outputs:
                if "WARNING" in output.get("text", ""):
                    print(
                        f"Removing Tensorflow warning in code cell {cell.execution_count}"
                    )
                    output["text"] = self.sub_warn.sub("", output.get("text", ""))

                if self.filter_all_stderr and output.get("name") == "stderr":
                    print(f"Removing stderr in code cell {cell.execution_count}")
                    continue

                pro_outputs.append(output)
            cell.outputs = pro_outputs
        return cell, resources


class RenumberCodeCellPreprocessor(preprocessors.Preprocessor):
    code_index = 0

    def preprocess(self, nb, resources):
        self.code_index = 0
        return super().preprocess(nb, resources)

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "code":
            # Renumber code cells from 1
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
    # Configuration
    # TODO: Move to environment variables or cmd args
    top_path = Path(".")
    format_code_cell = True     # Format code cells using black
    renumber_code_cell = True   # Renumber all code cells starting from 1
    set_default_kernel = True   # Set default kernel to "Python 3"

    ignore_checkpoints = True   # 
    overwrite_notebooks = False # Overwrite notebooks if True, otherwise save with ".mod"
    write_notebook = True       # Write formatted notebooks
    write_html = True           # Write formatted html

    preprocessor_list = [ClearWarningsPreprocessor]

    if set_default_kernel:
        preprocessor_list.append(SetKernelSpecPreprocessor)

    if format_code_cell:
        preprocessor_list.append(FormatCodeCellPreprocessor)

    if renumber_code_cell:
        preprocessor_list.append(RenumberCodeCellPreprocessor)

    # Specify our custom preprocessor
    c = Config()
    c.NotebookExporter.preprocessors = preprocessor_list
    c.HTMLExporter.preprocessors = preprocessor_list
    nb_exporter = NotebookExporter(c)
    html_exporter = HTMLExporter(c)
    # html_exporter.template_file = 'basic'

    # all_files = map(
    #     Path,
    #     ["demos/node-classification/gcn/gcn-cora-node-classification-example.ipynb"],
    # )
    all_files = top_path.glob("**/*.ipynb")

    # Go through all notebooks files in specified directory
    for file_loc in all_files:
        # Skip Modified files
        if "mod" in str(file_loc):
            continue
        # Skip checkpoints
        if ignore_checkpoints and ".ipynb_checkpoint" in str(file_loc):
            continue

        print(f"\nProcessing file {file_loc}")
        in_notebook = nbformat.read(str(file_loc), as_version=4)

        writer = writers.FilesWriter()

        if write_notebook:
            # Process the notebook to a new notebook
            (body, resources) = nb_exporter.from_notebook_node(in_notebook)

            # Write notebook file
            if overwrite_notebooks:
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

