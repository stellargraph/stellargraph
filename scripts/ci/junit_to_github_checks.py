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
import re
import sys
import textwrap
import xml.etree.ElementTree as ET


FILE_LINE = re.compile(r"([a-z0-9_/\\.]*):([0-9]*):")


def main():
    parser = argparse.ArgumentParser(
        description="convert a Pytest JUnit XML file to a checkstyle one, for reviewdog"
    )
    parser.add_argument("file", type=argparse.FileType("r"), default="-", nargs="?")
    args = parser.parse_args()

    tree = ET.parse(args.file)
    root = tree.getroot()

    invalid = []
    for testcase in root.findall(".//testcase"):
        for child in testcase:
            if child.tag in ("error", "failure"):
                classname = testcase.get("classname")
                name = testcase.get("name")

                match = FILE_LINE.search(child.text)
                if match is None:
                    invalid.append(
                        f"output of test '{name}' in '{classname}' does not contain match for filename & line regex /{FILE_LINE.pattern}/"
                    )
                    continue

                filename = match[1]
                line = int(match[2])

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
