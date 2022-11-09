# Contributing to Voxel

Thank you for considering to contribute to `voxel` and encouraging open-source tools for medical image analysis.

If you use Voxel in your work (research, company, etc.) and find it useful, spread the word!

## How to contribute
There are many ways to contribute to Voxel:

* Issues: Submitting bugs or suggesting new features
* Documentation: Adding to the documentation or to the examples
* Features: Implementing new features or bug fixes
* Community: Answering questions and helping others get started

## Submitting a new issue or feature request
Please do your best to follow these guidelines when opening an issue. It will make it signficantly easier to give useful feedback and resolve the issue faster.

### Found a bug?
We would very much appreciate if you could **make sure the bug was not already reported** (use the search bar on Github under Issues). If you cannot find you bug, follow the instructions in the [Bug Report](https://github.com/pyvoxel/pyvoxel/issues/new/choose) template.

### Want a new feature?

A world-class feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear about it!
  * Is it something you worked on and think could benefit the community? Awesome! Tell us what problem it solved for you.
2. Write a full paragraph describing the feature;
3. Provide a code snippet that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you post it. Follow the instructions in the [Feature Request](https://github.com/pyvoxel/pyvoxel/issues/new/choose)

## Contributing
Before writing code, we strongly advise you to search through the existing PRs or issues to make sure that nobody is already working on the same thing. If you are unsure, it is always a good idea to open an issue to get some feedback.

You will need basic git proficiency to be able to contribute to voxel. git is not the easiest tool to use but it has the greatest manual. Type git --help in a shell and enjoy. If you prefer books, [Pro Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [`repository`](https://github.com/pyvoxel/pyvoxel) by clicking on the 'Fork' button the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/pyvoxel.git
   $ cd pyvoxel
   $ git remote add upstream https://github.com/pyvoxel/pyvoxel.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment:

  ```bash
  pip install -e ".[dev]"
  ```

5. Develop features on your branch.

    As you work on the features, you should make sure that the test suite passes:

    ```bash
    $ make test
    ```

    After you make changes, autoformat them with:

    ```bash
    $ make autoformat
    ```


    If you modify documentation (`docs/source`), verify the documents build:

    ```bash
    $ make build-docs
    ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

HINT: Run all major formatting and checks using the following:

```bash
make autoformat test build-docs
```

### Checklist

1. Make the title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, mention the issue number in
  the pull request description
3. If your PR is a work in progress, start the title with `[WIP]`
4. Make sure existing tests pass;
5. Add high-coverage tests. Additions without tests will not be merged
6. All public methods must have informative docstrings in the google style.

### Tests

Library tests can be found in the
[tests folder](https://github.com/pyvoxel/pyvoxel/tree/master/tests).

From the root of the repository, here's how to run tests with `pytest` for the library:

```bash
$ make test
```

### Style guide
`pyvoxel` follows the [google style](https://google.github.io/styleguide/pyguide.html) for documentation.
