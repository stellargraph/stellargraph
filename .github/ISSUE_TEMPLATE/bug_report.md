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

*The following Python code can automatically compute the data for this section (for example: on macOS, copy the code and run `pbpaste | python -`):*

~~~python
import subprocess, platform, sys
packages=subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")
print("""Operating system: `{}`
Python version:
```
{}
```
Package versions:
<details>

```
{}
```

</details>""".format(platform.platform(), sys.version, packages))
~~~

### Additional context

Add any other context about the problem here. For example:

Data used: *if this occurs with a specific dataset, which dataset and how to acquire it*

Screenshots: *if applicable, add some screenshots to explain the behaviour*
