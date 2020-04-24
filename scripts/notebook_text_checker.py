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

import argparse
import nbformat
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# helpers

def cell_source(cell):
    return "".join(cell.source)

def cell_lines(cell, start, end):
    """
    Return the lines from `cell` that contain the range 'start:end' (for instance, a return value of `find`).
    """
    if start > end:
        raise ValueError(f"start ({start}) > end ({end})")

    source = cell_source(cell)
    start_line = source.rfind("\n", 0, start)
    end_line = source.find("\n", end)
    if end_line == -1:
        end_line = len(source)

    # 1-based line number of start of the section of lines
    start_num = source.count("\n", 0, start) + 1

    # we don't want to include either of the \n's: 'end' is already sliced exclusively, but 'start'
    # needs tweaking (note that start == -1 works correctly: that needs to slice from the start of
    # the string, and does exactly that source[0:end])
    return start_num, source[start_line + 1:end_line]

def message_with_line(cell, message, start=0, end=None):
    if end is None:
        end = start

    start_num, lines = cell_lines(cell, start, end)
    if "execution_count" in cell:
        cell_ref = f" [{cell.execution_count}]"
    else:
        cell_ref = ""

    indented = "".join(
        f"    {line_num:3} | {line}"
        for line_num, line in enumerate(lines.splitlines(True), start=start_num)
    )

    return f"{message}. Cell{cell_ref} lines:\n\n{indented}"

## checkers

CHECKERS = []

def checker(f):
    global CHECKERS
    CHECKERS.append(f)
    return f


NON_WHITESPACE_RE = re.compile(r"\S")
TITLE_RE = re.compile("^# .*")
HEADING_RE = re.compile("^(?P<level>##*) *")


@checker
def title_heading(notebook):
    first = notebook.cells[0]
    if first.cell_type != "markdown":
        return [message_with_line(first, "first cell should be a markdown cell containing the title")]

    source = cell_source(first)

    if TITLE_RE.fullmatch(source.strip()) is not None:
        # all good, only whitespace and the # ... title.
        return []

    title = TITLE_RE.search(source)
    if title is None:
        return [message_with_line(first, "first cell should contain a markdown title for the notebook `# ...`")]

    # report the error nicely
    first_trailing_nonwhitespace = NON_WHITESPACE_RE.search(source, title.end())
    if first_trailing_nonwhitespace is None:
        end = len(source)
    else:
        end = first_trailing_nonwhitespace.end()

    return [message_with_line(first, "first cell should contain only the markdown title for the notebook `# ...`", start=0, end=end)]


@checker
def other_headings(notebook):
    previous_heading_level = 1

    errors = []
    for cell in notebook.cells[1:]:
        if cell.cell_type != "markdown":
            continue

        source = cell_source(cell)
        for heading in HEADING_RE.finditer(source):
            if TITLE_RE.match(heading[0]):
                errors.append(message_with_line(cell, "only the first cell should contain a title heading `# ...`", start=heading.start()))

            level = len(heading["level"])
            if level > previous_heading_level + 1:
                previous = "#" * previous_heading_level
                errors.append(message_with_line(cell, f"headings should not skip levels; previous heading was H{previous_heading_level} `{previous} ...`, this heading is H{level}", start=heading.start()))

            previous_heading_level = level

    return errors


# ANSI terminal escape sequences
YELLOW_BOLD = "\033[1;33;40m"
LIGHT_RED_BOLD = "\033[1;91;40m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(
        description="Validates that the description in notebooks follows the expected message_with_line, so that they read consistently and render nicely."
    )
    parser.add_argument(
        "locations",
        nargs="+",
        help="Paths(s) to search for Jupyter notebooks to message_with_line",
    )

    args = parser.parse_args()

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

    all_errors = []
    for file_loc in all_files:
        # Skip checkpoints
        if ".ipynb_checkpoint" in str(file_loc):
            continue

        print(f"{YELLOW_BOLD}Checking file {file_loc}{RESET}")
        notebook = nbformat.read(str(file_loc), as_version=4)

        this_errors = []
        for checker in CHECKERS:
            errors = checker(notebook)
            for e in errors:
                print(f"{LIGHT_RED_BOLD}error{RESET}: {e}\n")
            this_errors.extend(errors)

        if errors:
            all_errors.append((file_loc, errors))

    if all_errors:
        # there was at least one problem!

        # try to annotate the build on buildkite with markdown
        def list_element(s):
            indented = textwrap.indent(s, "  ")
            # remove the indentation from the first line
            return f"- {indented[2:]}"

        def file_list(filename, errors):
            whole_list = "\n".join(list_element(error) for error in errors)
            return f"**`{filename}`**:\n\n{whole_list}"

        formatted = "\n\n".join(file_list(filename, errors) for filename, errors in all_errors)

        try:
            subprocess.run(
                [
                    "buildkite-agent",
                    "annotate",
                    "--style=error",
                    "--context=",
                    formatted,
                ]
            )
        except FileNotFoundError:
            # no agent, so probably on buildkite, and so silently no annotation
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
