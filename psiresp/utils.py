
import importlib


def update_dictionary(obj, key, value):
    if isinstance(value, dict):
        if not hasattr(obj.get(key), "__setitem__"):
            obj[key] = {}
        for k, v in value.items():
            update_dictionary(obj[key], k, v)
    else:
        obj[key] = value


def require_package(package, installation=""):
    installation_messages = {
        "rdkit": "conda install -c conda-forge rdkit",
        "psi4": "conda install -c psi4 psi4",
    }
    try:
        importlib.import_module(package)
    except ImportError:
        err = (
            f"This function requires the {package} package, "
            "but it could not be imported. "
            "Please make sure it is installed"
        )
        if not installation and package in installation_messages:
            installation = installation_messages[package]
        if installation:
            err += f", or install it with `{installation}`"
        raise ImportError(err) from None
