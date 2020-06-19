import yaml
import sys
import glob
from collections import Counter

MARKER = "# MARKER: list of all notebooks"
WORKFLOW = ".github/workflows/ci.yml"

def error(message, line):
    print(f"{WORKFLOW}:{line}: error: {message}", file=sys.stderr)
    sys.exit(1)

def main():
    with open(WORKFLOW) as f:
        contents = f.read()

    workflow = yaml.full_load(contents)
    try:
        marker_position = contents.index(MARKER)
    except:
        error(f"failed to find {MARKER!r} comment before the 'notebook:' matrix configuration", 0)

    marker_line = contents.count("\n", marker_position) + 1

    obj = workflow
    path = ["jobs", "notebooks", "strategy", "matrix", "notebook"]

    for i, key in enumerate(path):
        try:
            obj = obj[key]
        except KeyError:
            context = ".".join(repr(p) for p in path[:i])
            others = ", ".join(repr(k) for k in obj.keys())
            error(f"expected key {key!r} at {context}, found {others}", marker_line)


    # check for any notebooks listed more than once
    repeated = [name for name, count in Counter(obj).items() if count > 1]
    if repeated:
        repeated_str = ", ".join(repeated)
        error(f"found {len(repeated)} notebooks listed twice: {repeated_str}", marker_line)

    listed = set(obj)
    on_disk = set(glob.glob("demos/**/*.ipynb", recursive=True))

    if listed != on_disk:
        extra = listed - on_disk
        missing = on_disk - listed

        message = [f"found list of {len(listed)} notebooks in notebook testing step to be different to the {len(on_disk)} notebooks on disk"]

        if extra:
            extra_str = ", ".join(sorted(extra))
            message.append(f"notebooks listed but not on disk: {extra_str}")
        if missing:
            missing_str = ", ".join(sorted(missing))
            message.append(f"notebooks on disk but not listed: {missing_str}")

        error("; ".join(message), marker_line)

    print(f"{WORKFLOW}:{marker_line}: success: listed notebooks matches notebooks on disk")

if __name__ == "__main__":
    main()
