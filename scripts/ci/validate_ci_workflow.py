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

import yaml
import sys
import glob
from collections import Counter

NOTEBOOK_MARKER = "# MARKER: list of all notebooks"
NOTEBOOK_JOB = "notebooks"

REQUIRED_CHECK_MARKER = "# MARKER: list of all jobs"
REQUIRED_CHECK_JOB = "required-check"

WORKFLOW = ".github/workflows/ci.yml"


def error(message, line):
    print(f"::error file={WORKFLOW},line::{line}::{message}")
    sys.exit(1)


def find_marker_line(contents, marker):
    try:
        marker_position = contents.index(marker)
    except:
        error(
            f"failed to find {marker!r} comment before the 'notebook:' matrix configuration",
            0,
        )

    return contents.count("\n", 0, marker_position) + 1


def find_key(obj, path, marker_line):
    for i, key in enumerate(path):
        try:
            obj = obj[key]
        except KeyError:
            context = ".".join(repr(p) for p in path[:i])
            others = ", ".join(repr(k) for k in obj.keys())
            error(f"expected key {key!r} at {context}, found {others}", marker_line)

    return obj


def unique_and_equal(found, expected, name, step):
    # check for any notebooks listed more than once
    repeated = [name for name, count in Counter(found).items() if count > 1]
    if repeated:
        repeated_str = ", ".join(repeated)
        error(f"found {len(repeated)} {name} listed twice: {repeated_str}", marker_line)

    listed = set(found)

    if listed != expected:
        extra = listed - expected
        missing = expected - listed

        message = [
            f"found list of {len(listed)} {names} in '{step}' to be different to the {len(expected)} {name} on disk"
        ]

        if extra:
            extra_str = ", ".join(sorted(extra))
            message.append(f"{name} listed but not on disk: {extra_str}")
        if missing:
            missing_str = ", ".join(sorted(missing))
            message.append(f"{name} on disk but not listed: {missing_str}")

        error("; ".join(message), marker_line)

    print(f"{WORKFLOW}:{marker_line}: success: listed {name} matches {name} on disk")


def check_notebook_list(contents, workflow):
    marker_line = find_marker_line(contents, NOTEBOOK_MARKER)

    found = find_key(
        workflow, ["jobs", NOTEBOOK_JOB, "strategy", "matrix", "notebook"], marker_line
    )
    expected = set(glob.glob("demos/**/*.ipynb", recursive=True))

    unique_and_equal(
        found, expected, name="notebook(s)", step=NOTEBOOK_JOB,
    )


def check_needs_list(contents, workflow):
    marker_line = find_marker_line(contents, REQUIRED_CHECK_MARKER)

    jobs = find_key(workflow, ["jobs"], line)
    found = find_key(jobs, [REQUIRED_CHECK_JOB, "needs"], line)

    # this should depend on all of the other jobs...
    expected = set(jobs.keys())
    # ... except itself
    expected.remove(REQUIRED_CHECK_JOB)

    unique_and_equal(
        found, expected, name="job(s)", step=REQUIRED_CHECK_JOB,
    )


def main():
    with open(WORKFLOW) as f:
        contents = f.read()

    workflow = yaml.safe_load(contents)

    check_notebook_list(contents, workflow)
    check_needs_list(contents, workflow)


if __name__ == "__main__":
    main()
