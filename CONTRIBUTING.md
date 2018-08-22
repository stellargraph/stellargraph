# Contributing to Stellar-ML

Have you found a bug or have a new feature to suggest? Please read this before you start helping:

## Bug reporting

Please follow the following steps to report a bug:

1. First, be confident that the bug lies in Stellar-ML, not in your code or another package. Check the Stellar-ML FAQs.

1. The bug may already be fixed. Try updating to the latest version, and check the current and closed issues in GitHub. Search for similar issues and try and find if someone else has found the same bug already.

3. Make sure you provide us with useful information about your configuration: what OS are you using?

* 4. Provide a simple script that reproduces the bug. A bug that cannot be reproduced easily will most likely not be investigated.

5. Optionally, try and fix the bug and let us know how you go.

---

## Requesting a Feature

* 1. Provide a clear and detailed explanation of the feature you want and why it's important to add. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on library for Keras. It is crucial for Keras to avoid bloating the API and codebase.

* 2. Provide code snippets demonstrating the API you have in mind and illustrating the use cases of your feature. Of course, you don't need to write any real code at this point!

* 3. After discussing the feature you may choose to attempt a Pull Request. If you're at all able, start writing some code. We always have more work to do than time to do it. If you can write some code then that will speed the process along.

---

## Pull Requests

1. If your pull request will make a large change to the functionality of Stellar-ML it is best that you dicuss this first with the developers and the community. Please post a description of the changes to the Stellar-ML discussion board or as an issue in GitHub.

2. Start by checking out or updating the develop branch of Stellar-ML on GitHub. Create a new branch for your feature or bugfix named 'feature/XXX' or 'bugfix/XXX' where XXX is a short but descriptive name.

* 3. Make sure any new function or class you introduce has proper docstrings and documentation. Make sure any code you have changed also has updated Dostrings and documentation. Docstrings should follow the same style as the library. In particular, they should follow the Google docstring format ().

3. Make sure that any new features or bugfixes are tested by creating appropriate scripts in the tests directory. New features without relevant testing will not be approved.

4. Run the entire test suite by running `py.test tests/` in the top-level directory and ensure all tests pass. You will need to install the test requirements first: `pip install -e .[tests]`.

5. Ensure that your code is formmated by the Black style engine (). The automated tests include checking for Black formatted code.
 
6. When committing, use descriptive commit messages.

* 7. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

8. Create a pull request on GitHub from your branch to devleop. If you have already discussed the new features on GitHub with the developers and they are aware of what the pull request contains, then the developers will endeavour to approve the pull request promptly.

---

## Adding demos

We welcome new code that demonstrates the functionality of Stellar-ML on different datasets.