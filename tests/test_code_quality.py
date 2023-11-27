import pytest

import ml_project_2_mlp

from .utils import get_all_python_files, get_modules_and_functions

MODULES, FUNCTIONS = get_modules_and_functions(ml_project_2_mlp)
PYTHON_FILES = get_all_python_files(".")


@pytest.mark.parametrize("module", MODULES)
def test_check_module_docstring(module):
    """
    Tests that module has docstrings.
    """
    print()
    assert module.__doc__, f"Module {module.__name__} has no docstring."


@pytest.mark.parametrize("function", FUNCTIONS)
def test_check_function_docstring(function):
    """
    Tests that all functions have docstrings.
    """
    assert function.__doc__, f"Function {function.__name__} has no docstring."


@pytest.mark.parametrize("python_file", PYTHON_FILES)
def test_black_format(python_file):
    """
    Tests that all Python files are formatted with black.
    """
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
