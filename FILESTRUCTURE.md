This is an explanation of the file structure that the cookiecutter generated for you:

* Python source files:
  * The Python package source files are located in the `astrolink` directory.
  * `tests/test_astrolink.py` contains the unit tests for the package.
  * `tests/conftest.py` contains testing setup and configuration for `pytest`
  * The `notebooks` directory contains an example Jupyter notebook on how to use `astrolink`.
    This notebook is always executed during `pytest` execution and it is automatically
    rendered into the Sphinx documentation.
* Markdown files with meta information on the project. [Markdown](https://www.markdownguide.org/basic-syntax/) is
  a good language for these files, as it is easy to write and rendered into something beautiful by your git repository
  hosting provider.
  * `README.md` is the file that users will typically see first when discovering your project.
  * `COPYING.md` provides a list of copyright holders.
  * `LICENSE.md` contains the license you selected.
  * `TODO.md` contains a list of TODOs after running the cookiecutter. Following the
    instructions in that file will give you a fully functional repository with a lot
    of integration into useful web services activated and running.
  * `FILESTRUCTURE.md` describes the generated files. Feel free to remove this from the
    repository if you do not need it.
* Python build system files
  * `pyproject.toml` is the central place for configuration of your Python package.
    It contains the project metadata, setuptools-specific information and the configuration
    for your toolchain (like e.g. `pytest`).
  * `setup.py` is still required for editable builds, but you should not need to change it.
    In the future, `setuptools` will support editable builds purely from `pyproject.toml`
    configuration.
* Configuration for CI/Code Analysis and documentation services
  * `.github/workflows/ci.yml` describes the Github Workflow for Continuous
    integration. For further reading on workflow files, we recommend the
    [introduction into Github Actions](https://docs.github.com/en/free-pro-team@latest/actions/learn-github-actions/introduction-to-github-actions)
    and [the reference of available options](https://docs.github.com/en/free-pro-team@latest/actions/reference/workflow-syntax-for-github-actions).
  * `.github/dependabot.yml` configures the DependaBot integration on GitHub that
    allows you to automatically create pull requests for updates of the used actions
    in `.github/workflows/ci.yml`.
  * `.gitlab-ci.yml` describes the configuration for Gitlab CI. For further
    reading, we recommend [Gitlabs quick start guide](https://docs.gitlab.com/ee/ci/quick_start/)
    and the [Gitlab CI configuration reference](https://docs.gitlab.com/ce/ci/yaml/)
  * `.readthedocs.yml` configures the documentation build process at [ReadTheDocs](https://readthedocs.org).
    To customize your build, you can have a look at the [available options](https://docs.readthedocs.io/en/stable/config-file/v2.html).
