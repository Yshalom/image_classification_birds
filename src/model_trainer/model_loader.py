import os
import sys
from typing import Type, List, Tuple
import importlib.util
import torch.nn as nn

def _load_model_module(model_file_path: str):
    """
    Load a Python module from a file path.

    Args:
        model_file_path (str): Path to the Python file

    Returns:
        module: The loaded module
    """
    model_file_path = os.path.abspath(model_file_path)
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class_by_name(module, class_name: str) -> Type[nn.Module]:
    """
    Find a specific model class by name in a module.

    Args:
        module: The module to search in
        class_name (str): Name of the class to find

    Returns:
        Type[nn.Module]: The model class

    Raises:
        AttributeError: If class is not found
        TypeError: If class doesn't inherit from nn.Module
    """
    if hasattr(module, class_name):
        cls = getattr(module, class_name)
        if issubclass(cls, nn.Module):
            return cls
        else:
            raise TypeError(f"The class '{class_name}' does not inherit from nn.Module")
    else:
        raise AttributeError(f"Class '{class_name}' not found in module")


def _find_model_classes(module) -> List[Tuple[str, Type[nn.Module]]]:
    """
    Find all nn.Module subclasses in a module.

    Args:
        module: The module to search in

    Returns:
        List[Tuple[str, Type[nn.Module]]]: List of (name, class) tuples
    """
    model_classes = []
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, type) and issubclass(attr, nn.Module):
            model_classes.append((name, attr))
    return model_classes


def import_model_class(model_file_path: str, class_name: str | None = None) -> Type[nn.Module]:
    """
    Import a PyTorch model class from a Python file.

    Args:
        model_file_path (str): Path to the Python file containing the model class
        class_name (str, optional): Name of the class to import. If None, tries to find a class
                                   that inherits from nn.Module

    Returns:
        Type[nn.Module]: The model class
    """
    # Load the module
    module = _load_model_module(model_file_path)

    # Find the model class
    if class_name is not None:
        return _find_model_class_by_name(module, class_name)
    else:
        # Try to find a class that inherits from nn.Module
        model_classes = _find_model_classes(module)

        if len(model_classes) == 1:
            return model_classes[0][1]
        elif len(model_classes) == 0:
            raise ValueError(f"No nn.Module subclasses found in {model_file_path}")
        else:
            # Multiple classes found
            raise ValueError(f"Multiple model classes found: {[name for name, _ in model_classes]}")

def import_image_size(model_file_path: str) -> Tuple[int, int]:
    """
    Import the IMAGE_SIZE constant from a model file.

    Args:
        model_file_path (str): Path to the Python file containing the model class

    Returns:
        Tuple[int, int]: The image size tuple

    Raises:
        ValueError: If IMAGE_SIZE is not defined in the module or is not a valid tuple
    """
    module = _load_model_module(model_file_path)
    if not hasattr(module, "IMAGE_SIZE"):
        raise ValueError(f"IMAGE_SIZE not found in {model_file_path}")
    size = getattr(module, "IMAGE_SIZE")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError(f"IMAGE_SIZE must be a tuple of two ints, got {size}")
    return size
