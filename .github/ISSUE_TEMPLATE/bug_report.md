---
name: Bug report
about: Create a report to help us improve
labels: bug, sg-library

---

### Describe the bug

*A clear and concise description of what the bug is.*

### To Reproduce

Steps to reproduce the behavior:

1. *Go to '...'*
2. *Execute '...'*
3. *See error*

### Observed behavior

*Describe or paste the exact output that demonstrates the bug (for example, the stack trace if it is an exception).*

### Expected behavior

*A clear and concise description of what you expected to happen.*

### Environment

Operating system: *for example: macOS, Ubuntu*

Python version: *for example: 3.7.2*

Package versions: *for example: stellargraph==0.8.3, tensorflow==2.0.0*

*The following Python code can automatically compute the data for this section (for example: run the code in a Jupyter notebook cell, or, on macOS, copy the code and run `pbpaste | python -` in a terminal):*
~~~
import subprocess, platform, sys
pkgs = subprocess.Popen(["pip", "freeze"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode("utf-8")
print("Operating system: `{}`\nPython version:\n```\n{}\n```\nPackage versions:\n<details>\n\n```\n{}\n```\n\n</details>".format(platform.platform(), sys.version, pkgs))'
~~~

### Additional context

Add any other context about the problem here. For example:

Data used: *if this occurs with a specific dataset, which dataset and how to acquire it*

Screenshots: *if applicable, add some screenshots to explain the behaviour*
