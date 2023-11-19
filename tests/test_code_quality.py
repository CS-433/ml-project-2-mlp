import pytest
import pathlib

import ml_project_2_mlp.utils as utils

MODULES = [utils]
FUNCTIONS = [utils.check_import]


@pytest.mark.parametrize("module", MODULES)
def test_check_module_docstring(module):
    """
    Tests that module has docstrings.
    """
    assert module.__doc__, f"Module {module.__name__} has no docstring."


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_check_function_docstring(fn):
    """
    Tests that all functions have docstrings.
    """
    assert fn.__doc__, f"Function {fn.__name__} has no docstring."


def test_black_format():
    """
    Tests that all Python files are formatted with black.
    """
    root_path = pathlib.Path(__file__).parent.parent
    python_files = list(root_path.glob("**/*.py"))
    for python_file in python_files:
        content = python_file.read_text()
        try:
            import black
        except ModuleNotFoundError:
            assert False, "Please install black to check the formatting of your code."

        try:
            black.format_file_contents(content, fast=True, mode=black.FileMode())
            raise ValueError(f"{python_file.name} not formatted.")
        except black.NothingChanged:
            pass
