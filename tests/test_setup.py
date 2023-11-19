import sys
import toml
import importlib.metadata


def test_check_python_version():
    """
    Tests that the Python version is correct.
    """
    assert sys.version.startswith(
        "3.10.13"
    ), "Python version should be 3.10.13, but is {sys.version}."


def test_check_dependencies():
    """
    Tests that all dependencies are installed in the correct version.
    """
    # Load pyproject.toml
    with open("pyproject.toml", "r") as file:
        pyproject = toml.load(file)

    # Get the dependencies
    dependencies = pyproject["tool"]["poetry"]["dependencies"]

    # Check each dependency
    for package, version in dependencies.items():
        if package == "python":
            continue

        # Get the installed version
        installed_version = importlib.metadata.version(package)

        # Check if the installed version matches the specified version
        if isinstance(version, dict):
            assert (
                installed_version in version["version"]
            ), f"Version mismatch for {package}: expected {version}, got {installed_version}"
        elif isinstance(version, str):
            assert (
                installed_version in version
            ), f"Version mismatch for {package}: expected {version}, got {installed_version}"


def test_check_local_imports():
    """
    Checks that local imports can be resolved.
    """
    try:
        import ml_project_2_mlp

        ml_project_2_mlp.utils.check_import()
    except ModuleNotFoundError:
        err_msg = "The package could not be imported."
        raise ModuleNotFoundError(err_msg)
