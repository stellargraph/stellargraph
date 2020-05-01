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
import commonmark
import nbformat
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path


class FormattingError(Exception):
    """
    A checker that finds error(s) should raise this with all the errors.
    """

    def __init__(self, errors):
        if not isinstance(errors, list):
            errors = [errors]

        self.errors = errors


# notebook and cell helpers


def cell_source(cell):
    return "".join(cell.source)


def number_lines(lines, first_num):
    # indent the code (so it markdowns as a code block) and give it line numbers
    return "\n".join(
        f"    {line_num:3} | {line}"
        for line_num, line in enumerate(lines, start=first_num)
    )


COMMONMARK_PARSER = commonmark.Parser()


class MarkdownCell:
    """
    A markdown cell, that's been preprocessed (e.g. parsed into a Markdown AST).
    """

    def __init__(self, cell):
        source = cell_source(cell)
        self._lines = source.splitlines()

        self.ast = COMMONMARK_PARSER.parse(source)
        # make sure we don't have to deal with '<text><text>' elements
        self.ast.normalize()

    def lines(self, sourcepos):
        """
        Retrieve the lines corresponding to the 'sourcepos' range.

        Args:
            sourcepos: pair of pairs [[start line, start column], [end line, end column]], with one
                based lines and columns (the same as a commonmark's Node.sourcepos property)
        """
        one_based_start = sourcepos[0][0]
        # the commonmark sourcepos's are one based, so slicing needs an offset
        start = one_based_start - 1
        end = sourcepos[1][0] - 1

        # we want to include the end line
        lines = self._lines[start : end + 1]
        return number_lines(lines, one_based_start)


def parse_markdown_cells(notebook):
    def cell(c):
        if c.cell_type == "markdown":
            return MarkdownCell(c)
        return c

    return [cell(c) for c in notebook.cells]


def message_with_line(cell, message, sourcepos=[[1, 1], [1, 1]]):
    """
    Print 'message', along with some of the lines of 'cell'
    """
    return f"{message}. Some relevant lines from the cell:\n\n{cell.lines(sourcepos)}"


## commonmark helpers


def is_heading(elem):
    return elem.t == "heading"


def is_title(elem):
    return is_heading(elem) and elem.level == 1


def is_inline(elem):
    return elem.t in ("heading", "emph", "strong", "link", "image", "custom_inline")


def is_link(elem):
    return elem.t == "link"


def is_text(elem):
    return elem.t == "text"


SYNTAX_SUMMARY = {
    "block_quote": "> text",
    "code": "`code`",
    "emph": "*text*",
    "heading": "## text",
    "html_inline": "<tag>html</tag>",
    "image": "![text](url)",
    "item": "- text",
    "link": "[text](url)",
    "list": "- text",
    "strong": "**text**",
    "thematic_break": "---",
}


def syntax_summary(elem):
    """
    Return a basic summary of markdown syntax to as an example for users
    """
    return SYNTAX_SUMMARY.get(elem.t)


def direct_children(parent):
    """
    Iterate over the direct children of 'parent' (not all descendants, like 'parent.walker()')
    """
    elem = parent.first_child
    while elem is not None:
        yield elem
        elem = elem.nxt


def index_of_first(list_, pred):
    for i, x in enumerate(list_):
        if pred(x):
            return i

    return None


def closest_parent_sourcepos(elem):
    """
    Find the nearest defined sourcepos from a ancestor of elem.
    """
    while elem.sourcepos is None:
        elem = elem.parent

    return elem.sourcepos


def surrounding_sourcepos(elems, start_index, end_index=None):
    """
    Find a sourcepos that ranges from the last line of elems[start_index - 1] to the first line
    of elems[end_index + 1], handling the boundary conditions.
    """
    if end_index is None:
        end_index = start_index

    # chosen start element ...
    if start_index == 0:
        # ... is first, so start at its first line
        start = elems[start_index].sourcepos[0]
    else:
        # ... isn't first, so start at the previous element's last line
        start = elems[start_index - 1].sourcepos[1]

    # chosen end element ...
    if end_index == len(elems) - 1:
        # ... is last, so end at its last line
        end = elems[end_index].sourcepos[1]
    else:
        # ... isn't last, so end at the next element's first line
        end = elems[end_index + 1].sourcepos[0]

    return [start, end]


## checkers

CHECKERS = []


def checker(f):
    global CHECKERS
    CHECKERS.append(f)
    return f


@checker
def title_heading(cells):
    """
    The first cell should be the title (and only the title), so that the "cloud runner" cell appears
    immediately after it.
    """
    first = cells[0]
    if not isinstance(first, MarkdownCell):
        source = cell_source(first)
        # slice (not index) to handle an empty cell
        first_line = source.splitlines()[:1]
        lines = number_lines(first_line, 1)

        raise FormattingError(
            f"The first cell should be a markdown cell (containing only a title, like `# ...`). This one seems to be a code cell. First line of the cell:\n\n{lines}"
        )

    elems = list(direct_children(first.ast))
    title_idx = index_of_first(elems, is_title)

    if title_idx is None:
        # no title at at all
        raise FormattingError(
            message_with_line(
                first,
                "The first cell should be just the title for the notebook (like `# ...`) but the title seems to be missing here",
            )
        )

    if len(elems) == 1:
        # all good, only element is a title
        return

    # have a title, but there's other things too.
    sourcepos = surrounding_sourcepos(elems, title_idx)

    raise FormattingError(
        message_with_line(
            first,
            "The first cell should contain only the title (like `# ...`) for the notebook. Additional introductory content can be in a separate following cell",
            sourcepos=sourcepos,
        )
    )


@checker
def other_headings(cells):
    """
    No other H1/titles, and no heading level skipping.

    Extra titles break tables of contents, and heading level skipping causes Sphinx/reStructuredText
    warnings.
    """
    # keep track of any heading level skips, but only compare to headings that are nested correctly,
    # so an invalid section heading gets flagged, and so do any subheadings within that section.
    previous_valid_heading_level = 1
    first_invalid_heading_level = None

    errors = []
    for cell in cells[1:]:
        if not isinstance(cell, MarkdownCell):
            continue

        for elem, entering in cell.ast.walker():
            # only look at headings, and only look at them once
            if not is_heading(elem) or not entering:
                continue

            if is_title(elem):
                errors.append(
                    message_with_line(
                        cell,
                        "Found another title (like `# ...`) in internal cell. Later sections should use a high level heading (like `## ...` or `### ...`)",
                        sourcepos=elem.sourcepos,
                    )
                )

            if elem.level > previous_valid_heading_level + 1:
                previous = "#" * previous_valid_heading_level

                if first_invalid_heading_level is None:
                    first_invalid_heading_level = elem.level

                # assume that there's only one level skip (e.g. H1, H3, H3(*), H4, H3), and that the
                # relative levels within the invalid section are correct. This means for all H3
                # suggest only H2, but for H4 suggest H3 too (to continue nesting within (*)).
                levels_from_first_invalid = elem.level - first_invalid_heading_level
                max_suggestion_level = (
                    previous_valid_heading_level + levels_from_first_invalid + 1
                )

                suggestions = ", ".join(
                    f"`{'#' * i} ...`" for i in range(2, max_suggestion_level + 1)
                )
                errors.append(
                    message_with_line(
                        cell,
                        f"Found a heading H{elem.level} that skips level(s) from most recent valid heading (H{previous_valid_heading_level} `{previous} ...`). Consider using: {suggestions}",
                        sourcepos=elem.sourcepos,
                    )
                )
            else:
                # this was valid, so we can reset our counts
                previous_valid_heading_level = elem.level
                first_invalid_heading_level = None

    if errors:
        raise FormattingError(errors)


@checker
def simple_inline_formatting(cells):
    """
    rST doesn't easily supported nested formatting, such as Markdown like [some `code` within a
    link](...) or **`bold code`**, so we disallow it.

    http://docutils.sourceforge.net/FAQ.html#is-nested-inline-markup-possible
    """

    errors = []
    for cell in cells:
        if not isinstance(cell, MarkdownCell):
            continue

        for elem, entering in cell.ast.walker():
            if not entering:
                # only look at things once
                continue

            if not is_inline(elem):
                # not an inline formatting, so not relevant
                continue

            if all(is_text(child) for child in direct_children(elem)):
                # if all of the children are plain text, this is perfect!
                continue

            # an inline element that contains non-text elements, error!
            summary = syntax_summary(elem)
            if summary is None:
                summary = ""
            else:
                summary = f" (`` {summary} ``)"

            suggestions = ["removing the some of the formatting"]
            if is_link(elem):
                suggestions.append(
                    f"placing the link separately (like `<text> ([link](<url>))` or `<text> ([docs](<url>))`)"
                )

            errors.append(
                message_with_line(
                    cell,
                    f"Found some nested formatting within a {elem.t} element{summary}, which isn't supported in reStructuredText, as used by Sphinx and Read the Docs. Consider: {'; '.join(suggestions)}",
                    sourcepos=closest_parent_sourcepos(elem),
                )
            )

    if errors:
        raise FormattingError(errors)


# ANSI terminal escape sequences
YELLOW_BOLD = "\033[1;33;40m"
LIGHT_RED_BOLD = "\033[1;91;40m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Validates that the descriptions in notebooks follow the expected format, so that the notebooks read consistently and render nicely."
    )
    parser.add_argument(
        "locations",
        nargs="+",
        help="Paths(s) to search for Jupyter notebooks to check",
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
        cells = parse_markdown_cells(notebook)

        this_errors = []
        for checker in CHECKERS:
            try:
                checker(cells)
            except FormattingError as exc:
                for e in exc.errors:
                    print(f"{LIGHT_RED_BOLD}error{RESET}: {e}\n")
                this_errors.extend(exc.errors)

        if this_errors:
            all_errors.append((file_loc, this_errors))

    if all_errors:
        # there was at least one problem!

        # try to annotate the build on buildkite with markdown
        def list_element(s):
            indented = textwrap.indent(s, "  ")
            # remove the indentation from the first line
            return f"- {indented[2:]}"

        def render_path(path):
            text = f"**`{path}`**"

            # if the commit for the build is known, include a link to that exact rendered notebook,
            # for convenience
            try:
                commit = os.environ["BUILDKITE_COMMIT"]
            except KeyError:
                pass
            else:
                url = f"https://nbviewer.jupyter.org/github/stellargraph/stellargraph/blob/{commit}/{path}"
                text = f"{text} ([rendered notebook]({url}))"

            return text

        def file_list(path, errors):
            whole_list = "\n".join(list_element(error) for error in errors)
            return f"{render_path(path)}:\n\n{whole_list}"

        file_lists = "\n\n".join(file_list(path, errors) for path, errors in all_errors)

        command = f"python {__file__} demos/"
        formatted = f"""\
Found some notebooks with inconsistent formatting. These notebooks may be less clear or render incorrectly on Read the Docs. Please adjust them.

{file_lists}

This check can be run locally, via `{command}`."""

        try:
            subprocess.run(
                [
                    "buildkite-agent",
                    "annotate",
                    "--style=error",
                    "--context=notebook_text_checker",
                    formatted,
                ]
            )
        except FileNotFoundError:
            # no agent, so probably not on buildkite, and so silently continue without an annotation
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
