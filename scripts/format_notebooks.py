#!/usr/bin/env python3

import nbformat
import re

from traitlets import Set, Integer, Bool
from traitlets.config import Config
from pathlib import Path
from nbconvert import NotebookExporter, HTMLExporter, writers, preprocessors


class ClearWarningsPreprocessor(preprocessors.Preprocessor):
    remove_metadata_fields = Set(
        {'collapsed', 'scrolled'}
    ).tag(config=True)

    code_index = 0

    filter_all_stderr = Bool(True,
        help="Remove all stderr outputs."
    ).tag(config=True)

    renumber_code_cells = Bool(False,
        help="Renumber code cells consistently starting from 1. "
        "Note this will erase the true execution order, so be careful!"
    ).tag(config=True)


    def preprocess(self, nb, resources):
        # Set the default kernel:
        if "kernelspec" in nb.metadata and nb.metadata["kernelspec"]['name'] != "python3":
            print("Incorrect kernelspec:", nb.metadata["kernelspec"])

        nb.metadata["kernelspec"] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

        return super().preprocess(nb, resources)

    def preprocess_cell(self, cell, resources, cell_index):
        sub_warn = re.compile(r"^WARNING:tensorflow.*\n.*\n.*\n", re.MULTILINE)

        if cell.cell_type == 'code':
            # Renumber code cells from 1
            self.code_index += 1
            if self.renumber_code_cells:
                cell.execution_count = self.code_index

            # Search and remove warnings in outputs
            pro_outputs = []
            for output in cell.outputs:
                if "WARNING" in output.get("text",""):
                    print(f"Removing Tensorflow warning in code cell {cell.execution_count}")
                    output["text"] = sub_warn.sub("", output.get("text",""))
                
                if self.filter_all_stderr and output.get("name") == "stderr":
                    print(f"Removing stderr in code cell {cell.execution_count}")
                    continue

                pro_outputs.append(output)
            cell.outputs = pro_outputs
                    
        return cell, resources

if __name__=="__main__":
    # Find notebooks to process
    top_path = Path(".")
    ignore_checkpoints = True
    overwrite_notebooks = True
    write_notebook = True
    write_html = True

    # Go through all notebooks files in specified directory
    for file_loc in top_path.glob('**/*.ipynb'):
        if "mod" in str(file_loc):
            continue
        # Skip checkpoints
        if ignore_checkpoints and ".ipynb_checkpoint" in str(file_loc):
            continue

        print(f"\nProcessing file {file_loc}")
        in_notebook = nbformat.read(str(file_loc), as_version=4)

        # Specify our custom preprocessor
        c =  Config()
        c.NotebookExporter.preprocessors = [ClearWarningsPreprocessor]
        c.HTMLExporter.preprocessors = [ClearWarningsPreprocessor]
        nb_exporter = NotebookExporter(c)

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

        html_exporter = HTMLExporter(c)
        #html_exporter.template_file = 'basic'

        if write_html:
            # Process the notebook to HTML
            (body, resources) = html_exporter.from_notebook_node(in_notebook)

            html_file_loc = str(file_loc.with_suffix(""))
            print(f"Writing HTML to {html_file_loc}.html")
            writer.write(body, resources, html_file_loc)
