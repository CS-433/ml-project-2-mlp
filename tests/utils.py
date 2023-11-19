import inspect
import pathlib
import pkgutil


def get_modules_and_functions(package: str) -> tuple[list, list]:
    """
    Returns all (sub-)modules and functions in a package.
    """
    all_modules = []
    all_functions = []

    def recurse(package):
        # Iterate through all the modules in the package
        for _, modname, ispkg in pkgutil.iter_modules(
            package.__path__, package.__name__ + "."
        ):
            if modname not in all_modules:  # Avoid duplicates
                all_modules.append(modname)

                # Import the module
                module = __import__(modname, fromlist="dummy")

                # Iterate through all the functions in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj):
                        all_functions.append(f"{modname}.{name}")

                # Recurse if it's a package
                if ispkg:
                    recurse(module)

    recurse(package)

    return all_modules, all_functions


def get_all_python_files(path: str) -> pathlib.Path:
    """
    Returns all Python files recursively in a directory.

    Args:
        path: path to the directory

    Returns:
        list[pathlib.Path]
    """
    return pathlib.Path(path).glob("**/*.py")
