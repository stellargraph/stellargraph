## StellarGraph Library Release Procedure

1. **Create release branch**

   - Create the release branch
     ```shell
     git checkout -b release-X.X.X develop
     ```
   - Increase the version and apply other release-related changes
     - MUST do:
       - Version bumping: Change version from “X.X.Xb” to “X.X.X”. E.g. version=”0.2.0b” to version=”0.2.0”
         - `stellargraph/version.py`
         - `meta.yaml`
       - Update Changelog section header and "Full Changelog" link to point to specific version tag instead of `HEAD`. Note: these links will be broken until the tag is pushed later.
     - CAN do:
       - Minor bug fixes if necessary
     - NEVER do:
       - Add new features

2. **Merge release branch into `master` locally**

    This step gets your local `master` branch into release-ready state.

    ```shell
    git checkout master
    git merge --no-ff release-X.X.X -m "Release X.X.X"
    git tag -a vX.X.X -m "Release X.X.X"
    ```

3. **Upload to PyPI**

    NOTE: An account on PyPI is required to upload - create an account and ask a team member to add you to the organisation.

   - Install build/upload requirements:
     ```shell
     pip install wheel twine
     ```
   - Build distribution files:
     ```shell
     python setup.py sdist bdist_wheel
     ```
     This will create files `stellargraph-<version>.tar.gz` and `stellargraph-<version>-py3-none-any.whl` under the `dist` directory.
   - Upload to PyPi
     ```shell
     twine upload dist/stellargraph-<version>*
     ```
   - Check upload is successful: https://pypi.org/project/stellargraph/

4. **Upload to Conda Cloud**

   NOTE: An account on Conda Cloud is required to upload - create an account and ask a team member to add you to the organization.

   NOTE: These instructions are taken from https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-packages/)

   - Turn off auto-uploading
     ```shell
     conda config --set anaconda_upload no
     ```
   - Build package
     ```shell
     conda build .
     ```
   - Upload to Anaconda Cloud in the “stellargraph” organization
     ```shell
     conda build . --output # find the path to the package
     anaconda login
     anaconda upload -u stellargraph /path/to/conda-package.tar.bz2
     ```

5. **Make release on GitHub**

    After successfully publishing to PyPi and Conda, we now want to make the release on GitHub.

   - Temporarily turn off branch protection on the `master` branch. Ask a team member if you are unsure.
   - Push `master` branch
     ```shell
     git push --follow-tags origin master
     ```
   - Turn branch protection back on.
   - Go to the tags on the GitHub stellargraph homepage: https://github.com/stellargraph/stellargraph/tags
   - Next to the release tag, click the “...” button and select “create release”
   - Add the title and text of the metadata: a title “Release X.X.X” and text copied from the changelog is good practice
   - Click “Publish release”

6. **Get `develop` into correct state for next development version**

    We want the merge any of the changes made during the release back into `develop`, and make sure the new version in `develop` is correct.

   - Switch to `develop` branch:
     ```shell
     git checkout develop
     ```
   - Increase the version: in `stellargraph/version.py`, change version from `X.X.X` to `X.X+1.Xb`. E.g. `__version__ = "0.2.0"` to `__version__ = "0.3.0b"`. (To stay consistent we use `b` to indicate “beta”, but python will accept any string after the number. In semantic versioning: first number for major release, second number for minor release, third number for hotfixes.)
     ```shell
     git add stellargraph/version.py
     git commit -m "Bump version"
     ```
   - Merge `master` into `develop` and resolve conflict by using the new version in `develop`:
     ```shell
     git merge master
     ```
   - Temporarily turn off branch protection on the `develop` branch. Ask a team member if you are unsure.
   - Push the merge commit (and the version change):
     ```shell
     git push origin develop
     ```
   - Turn branch protection back on.


## More Information

Gitflow Examples:
https://gitversion.readthedocs.io/en/latest/git-branching-strategies/gitflow-examples/
