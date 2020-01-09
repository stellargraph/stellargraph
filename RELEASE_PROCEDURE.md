## StellarGraph Library Release Procedure

1. **Create release branch**

   - Create the release branch
     ```shell
     git checkout -b release-X.X.X develop
     ```
   - Increase the version and apply other release-related changes
     - MUST do:
       - Version bumping: In stellargraph/version.py, change version from “X.X.Xb” to “X.X.X”. E.g. version=”0.2.0b” to version=”0.2.0”
       - Other release-related changes e.g. changelog update
     - CAN do:
       - Minor bug fixes if necessary
     - NEVER do:
       - Add new features

2. **Merge release branch into `master`**

   ```shell
   git checkout master
   git merge --no-ff release-X.X.X -m "Release X.X.X"
   git tag -a vX.X.X -m "Release X.X.X"
   git push --follow-tags origin master
   ```

3. **Merge `master` into `develop`**

   - Do the merge
     ```shell
     git checkout develop
     git merge master
     ```
   - Increase the version: in `stellargraph/version.py`, change version from `X.X.X` to `X.X+1.Xb`. E.g. `__version__ = "0.2.0"` to `__version__ = "0.3.0b"`. (To stay consistent we use `b` to indicate “beta”, but python will accept any string after the number. In semantic versioning: first number for major release, second number for minor release, third number for hotfixes.)
   - Commit the version change:
     ```shell
     git commit -am "Bumped version"
     ```
   - Push the merge commit (and the version change):
     ```shell
     git push origin develop
     ```

4. **Add GitHub release metadata**

   - Go to the tags on the GitHub stellargraph homepage: https://github.com/stellargraph/stellargraph/tags
   - Next to the release tag, click the “...” button and select “create release”
   - Add the title and text of the metadata: a title “Release X.X.X” and text copied from the changelog is good practice
   - Click “Publish release”

5. **Upload new version to PyPI** (NOTE: An account on PyPI is required to upload; talk to Denis or Alex.)

   - Install build/upload requirements:
     ```shell
     pip install wheel twine
     ```
   - Update to master branch on local checkout:
     ```shell
     git checkout master
     git pull
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

6. **Upload to Conda Cloud** (these instructions are taken from https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-packages/)
   - Update the version number in build.yaml
   - Turn off auto-uploading
     ```shell
     conda config --set anaconda_upload no
     ```
   - Build package
     ```shell
     conda build . --output
     ```
   - Upload to Anaconda Cloud in the “stellargraph” organization
     ```shell
     anaconda login
     anaconda upload -u stellargraph /path/to/conda-package.tar.bz2
     ```

## More Information

Gitflow Examples:
https://gitversion.readthedocs.io/en/latest/git-branching-strategies/gitflow-examples/
