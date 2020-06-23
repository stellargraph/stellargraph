# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import defaultdict
import os
import sys
import textwrap
import xml.etree.ElementTree as ET


def deduce_file_position(testcase, base):
    name = testcase.get("name")
    try:
        index = name.index("[")
    except:
        # no parameters
        base_name = name
    else:
        base_name = name[:index]

    classname = testcase.get("classname")
    filename = classname.replace(".", "/") + ".py"

    open_filename = os.path.join(base, filename)
    try:
        with open(open_filename) as f:
            contents = f.read()
    except:
        raise ValueError(
            f"failed to read '{filename}', as deduced from failing test '{name}' in '{classname}'"
        )

    # estimate the location of the test by doing a string search
    try:
        index = contents.index(f"def {base_name}")
    except:
        raise ValueError(f"could not find failing test '{base_name}' in '{filename}'")

    # number of newlines since the start of the file = zero-based line, GitHub uses one-based lines
    line = contents.count("\n", 0, index) + 1

    return filename, line


def main():
    parser = argparse.ArgumentParser(
        description="convert a Pytest JUnit XML file to a checkstyle one, for reviewdog"
    )
    parser.add_argument("file", type=argparse.FileType("r"), default="-", nargs="?")
    args = parser.parse_args()

    base_directory = os.path.join(os.path.dirname(__file__), "../..")

    tree = ET.parse(args.file)
    root = tree.getroot()

    invalid = []
    for testcase in root.findall(".//testcase"):
        for child in testcase:
            if child.tag in ("error", "failure"):
                try:
                    filename, line = deduce_file_position(testcase, base_directory)
                except ValueError as e:
                    invalid.append(e)
                    continue

                name = testcase.get("name")
                base_message = child.get("message").replace("\\n", "\n")
                indented = textwrap.indent(base_message, "    ")

                message = f"""\
Test '{name}' failed:

{indented}
"""
                # multiline output is possible by escaping (only) the \n
                # https://github.com/actions/toolkit/issues/193#issuecomment-605394935
                encoded = message.replace("\n", "%0A")
                print(f"::error file={filename},line={line}::{encoded}")

    if invalid:
        print(
            "failed to understand some failed test(s) (fixing the test(s) will stop this error too):",
            file=sys.stderr,
        )
        for exc in invalid:
            print(f"- {exc}", file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()
