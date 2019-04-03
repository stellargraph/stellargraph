# Contributing to StellarGraph

Have you found a bug or have a new feature to suggest? Please read this before you start helping:

## Bug reporting

Please follow these steps to report a bug:

1. First, be confident that the bug lies in StellarGraph, not in your code or another package. <!-- Check the StellarGraph FAQs. -->

1. The bug may already be fixed. Try updating to the latest version, and check the current and closed issues in GitHub. Search for similar issues and try and find if someone else has found the same bug already.

3. Make sure you provide us with useful information about your configuration: what OS are you using?

4. Provide a simple script that reproduces the bug. A bug that cannot be reproduced easily will most likely not be investigated.

5. Optionally, try and fix the bug and let us know how you go.

## Contributing Code

### Contributor License Agreement

In order to contribute to StellarGraph, please ensure that you have signed a Contributor License Agreement (CLA). You will 
be guided through the process of digitally signing our CLA when you create a pull request. 

### Be Friendly ###
 
StellarGraph considers courtesy and respect for others an essential part of the community, and we strongly encourage everyone to be friendly when engaging with others. Please be helpful when people are asking questions, and on technical disagreements ensure that the issues are discussed in a respectful manner.

### Proposing a new feature

1. Give a clear and detailed explanation of the feature and why it should be added. This is best done by creating an issue on [GitHub](https://github.com/stellargraph/stellargraph/issues) addressing the new feature. Propose a clear API for using the feature, preferably with a small snippet of pseudo-code.

2. If this is an implementation of an algorithm in the literature, please give a link to a paper describing the algorithm.

3. If you choose to implement the feature you can do so by forking the StellarGraph repository and creating a new branch addressing your feature from the `develop` branch. After writing code implementing the feature in this branch make a Pull Request to the `develop` branch of the main StellarGraph repository. See below for more details on submitting your pull request.

### Adding demos

1. We welcome new code that demonstrates the functionality of StellarGraph on different datasets. To add a demo it is best to give a clear and detailed explanation of the demo. This is best done by creating an issue addressing the new demo.

3. Describe the dataset that you are using, how it can be downloaded, and the licence conditions. Please don't put the dataset in the GitHub repository.

2. If this demo is replicating an experiment in the literature, please give a link to a paper describing the algorithm.

3. If you choose to implement the demo you can do so by forking the StellarGraph repository and creating a new branch from the `develop` branch. Put the code for the demo in the `demos` directory and make a Pull Request to the `develop` branch of the main StellarGraph repository. See the next section for more details on submitting your pull request.


### Pull Requests

1. If your pull request will make a large change to the functionality of StellarGraph it is best that you discuss this first with the developers and the community. Please post a description of the changes to the StellarGraph as an issue in GitHub.

2. Start checking out or updating the develop branch of StellarGraph on GitHub. Create a new branch for your feature or bugfix named 'feature/XXX' or 'bugfix/XXX' where XXX is a short but descriptive name.

3. Make sure that any new features or bugfixes are tested by creating appropriate scripts in the tests directory. New features without relevant testing will not be approved.

4. Run the entire test suite by running `py.test tests/` in the top-level directory and ensure all tests pass. You will need to install the test requirements first: `pip install -e .[tests]`.

5. Ensure that any new function or class you introduce has proper docstrings and documentation. Make sure any code you have changed also has updated dostrings and documentation. Docstrings should follow the same style as the library, we follow the Google style (https://github.com/google/styleguide/blob/gh-pages/pyguide.md), examples of Google style docstrings can be found [here](http://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google).

6. All code in StellarGraph is formatted using the Black style engine (https://github.com/ambv/black). The automated tests include checking for Black formatted code, so make sure that you run black on all your code before submitting a pull request.

7. When committing, use descriptive commit messages.

8. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

9. Create a pull request on GitHub from your branch to `develop` branch. If you have already discussed the new features on GitHub with the developers and they are aware of what the pull request contains, then the developers will endeavour to approve the pull request promptly.
