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

    with open(os.path.join(base, filename)) as f:
        contents = f.read()

    # estimate the line of the test by doing a string search
    try:
        index = contents.index(f"def {base_name}")
    except:
        raise ValueError(f"count not find failing test '{base_name}' in {filename}")

    line = contents.count("\n", index) + 1
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

    tests_per_file = defaultdict(list)
    invalid = []

    for testcase in root.findall(".//testcase"):
        children = {child.tag: child for child in testcase}

        problem_child = children.get("error") or children.get("failure")

        if problem_child is not None:
            try:
                filename, line = deduce_file_position(testcase, base_directory)
            except ValueError as e:
                invalid.append(e)
                continue

            tests_per_file[filename].append(
                (line, testcase.get("name"), problem_child.get("message"))
            )

    checkstyle = ET.Element("checkstyle")
    for file_name, tests in tests_per_file.items():
        this_file = ET.Element("file", name=file_name)

        for (line, test_name, message) in tests:
            real_newlines = message.replace('\\n', '\n')
            formatted = f"""Test {test_name} failed:

{real_newlines}"""

            error = ET.Element(
                "error",
                severity="error",
                line=str(line),
                source=test_name,
                message=formatted,
            )
            this_file.append(error)

        checkstyle.append(this_file)

    ET.ElementTree(checkstyle).write(sys.stdout, encoding="unicode")

    if invalid:
        print(
            "failed to understand several failed tests (fixing the tests will stop this error too):",
            file=sys.stderr,
        )
        for exc in invalid:
            print(f"- {exc}", file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()
