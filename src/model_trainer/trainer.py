"""
Model trainer program for bird species image classification.

This program:
- Gets a path to a Python file that has a PyTorch class representing a model
- Imports the class from the model
- Loads the bird database using @src/database_reader/ package
- Creates cached image tensors for efficient training
- In a loop of i in [1, ..., 10]:
  + Creates instance of the model
  + Trains it with ADAM optimizer and cross-entropy loss
  + Does 300 training loops
  + After every 30 training loops, evaluates loss with 'train', 'test' and 'validation' databases
  + Plots the loss CSV file in a subdirectory 'log/train-i.csv' (under the model file)
  + Saves the model weights at "weights/model-i.pt"
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type, List, Tuple
import importlib.util

# Add src to path to import from database_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database_reader.bird_database import BirdDatabase
from image_cache import ImageCache
from constants import DB_TEST_PATH, DB_TRAIN_PATHS, DB_VALIDATION_PATH, README_PATH, LABEL_NAME_PATH

DTYPE = torch.uint8
TRAINING_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CACHE_DEVICE = torch.device("cpu")
BATCH_SIZE = 1024
TRAINING_EPOCHS = 30
LOGGING_INTERVAL = 3
AMOUNT_OF_MODELS = 2
LEARNING_RATE = 0.001

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

def evaluate_model(model: nn.Module, image_cache: ImageCache) -> float:
    """
    Evaluate the model on a database and return average loss.

    Args:
        model: The PyTorch model to evaluate
        image_cache: ImageCache instance containing pre-loaded images

    Returns:
        float: Average loss over the database
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = min(BATCH_SIZE, len(image_cache))
        num_batches = (len(image_cache) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_cache))

            # Get batch from cache
            batch_images, batch_labels = image_cache[start_idx:end_idx]

            # Move to device, convert type and normalize
            batch_images = batch_images.to(TRAINING_DEVICE).to(model.input_dtype) / 255
            # Move to device
            batch_labels = batch_labels.to(TRAINING_DEVICE)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item() * len(batch_images)

    return total_loss / len(image_cache)


def _prepare_training_directories(model_file_path: str) -> tuple[str, str]:
    """
    Prepare directories for saving logs and weights.

    Args:
        model_file_path: Path to the original model file

    Returns:
        Tuple of (log_dir, weights_dir)
    """
    model_dir = os.path.dirname(os.path.abspath(model_file_path))
    log_dir = os.path.join(model_dir, 'log')
    weights_dir = os.path.join(model_dir, 'weights')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    return log_dir, weights_dir


def _create_log_file(log_dir: str, model_id: int) -> str:
    """
    Create and initialize the CSV log file.

    Args:
        log_dir: Directory for log files
        model_id: ID of this model instance

    Returns:
        Path to the log file
    """
    log_file = os.path.join(log_dir, f'train-{model_id}.csv')
    with open(log_file, 'w') as f:
        f.write("training loops, loss(DB-train), loss(DB-test), loss(DB-validation)\n")
    return log_file


def _train_epoch(model: nn.Module,
                 train_cache: ImageCache,
                 optimizer: optim.Adam,
                 criterion: nn.CrossEntropyLoss) -> None:
    """
    Train for one epoch.

    Args:
        model: The PyTorch model to train
        train_cache: ImageCache containing training images
        optimizer: Optimizer to use
        criterion: Loss function
    """
    # Sample a random batch
    indices = torch.randperm(len(train_cache))

    batch_size = min(BATCH_SIZE, len(train_cache))
    num_batches = (len(train_cache) + batch_size - 1) // batch_size

    model.train()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(train_cache))

        # Get batch from cache
        batch_images, batch_labels = train_cache[indices[start_idx:end_idx]]

        # Move to device, convert type and normalize
        batch_images = batch_images.to(TRAINING_DEVICE).to(model.input_dtype) / 255
        # Move to device
        batch_labels = batch_labels.to(TRAINING_DEVICE)

        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()


def _evaluate_and_log(model: nn.Module,
                      train_cache: ImageCache,
                      test_cache: ImageCache,
                      val_cache: ImageCache,
                      epoch: int,
                      log_file: str) -> None:
    """
    Evaluate model and log results.

    Args:
        model: The PyTorch model to evaluate
        train_cache: ImageCache containing training images
        test_cache: ImageCache containing test images
        val_cache: ImageCache containing validation images
        epoch: Current epoch number
        log_file: Path to log file
    """
    model.eval()
    avg_train_loss = evaluate_model(model, train_cache)
    avg_test_loss = evaluate_model(model, test_cache)
    avg_val_loss = evaluate_model(model, val_cache)

    # Log to CSV
    with open(log_file, 'a') as f:
        f.write(f"{epoch},{avg_train_loss:.6f},{avg_test_loss:.6f},{avg_val_loss:.6f}\n")

    print(f"Epoch [{epoch}], "
        f"Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, "
        f"Val Loss: {avg_val_loss:.6f}")


def _save_model_weights(model: nn.Module,
                        weights_dir: str,
                        model_id: int) -> None:
    """
    Save model weights to file.

    Args:
        model: The PyTorch model to save
        weights_dir: Directory to save weights
        model_id: ID of this model instance

    Returns:
        Path to the saved weights file
    """
    weights_file = os.path.join(weights_dir, f'model-{model_id}.pt')
    torch.save(model.state_dict(), weights_file)
    print(f"Model {model_id} weights saved to {weights_file}")


def train_model(model: nn.Module,
                train_cache: ImageCache,
                test_cache: ImageCache,
                val_cache: ImageCache,
                log_file: str):
    """
    Train a single model instance.

    Args:
        model: The PyTorch model to train
        train_db: Training database
        test_db: Testing database
        val_db: Validation database
        log_file: Path to the log file
    """
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(1, TRAINING_EPOCHS + 1):
        _train_epoch(model, train_cache, optimizer, criterion)

        # Evaluate and log every several epochs
        if epoch % LOGGING_INTERVAL == 0 or epoch == TRAINING_DEVICE or epoch == 1:
            _evaluate_and_log(model, train_cache, test_cache, val_cache, epoch, log_file)


def main():
    """Main function to run the model training."""
    if len(sys.argv) < 2:
        print("Usage: python trainer.py <model_file_path> [model_class_name]")
        print("Example: python trainer.py ../model_trainer/simple_cnn.py SimpleCNN")
        sys.exit(1)

    model_file_path = sys.argv[1]
    class_name = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        # Import the model class
        print(f"Importing model from {model_file_path}")
        ModelClass = import_model_class(model_file_path, class_name)
        print(f"Successfully imported model class: {ModelClass.__name__}")

        # Initialize databases
        print("Loading bird databases...")
        test_db = BirdDatabase(
            db_path=DB_TEST_PATH,
            readme_path=README_PATH,
            label_name_path=LABEL_NAME_PATH
        )
        val_db = BirdDatabase(
            db_path=DB_VALIDATION_PATH,
            readme_path=README_PATH,
            label_name_path=LABEL_NAME_PATH
        )
        train_db = BirdDatabase(
            db_path=DB_TRAIN_PATHS,  # Uses list of paths
            readme_path=README_PATH,
            label_name_path=LABEL_NAME_PATH
        )

        print(f"Loaded datasets:", f"Train: {len(train_db)}", f"Test: {len(test_db)}", f"Validation: {len(val_db)}", sep="\n\t")

        # Create image caches for efficient training
        print(f"Creating image caches ...")
        # Import the IMAGE_SIZE constant from the model file
        image_size = import_image_size(model_file_path)
        test_cache = ImageCache(test_db, image_size, CACHE_DEVICE)
        val_cache = ImageCache(val_db, image_size, CACHE_DEVICE)
        train_cache = ImageCache(train_db, image_size, CACHE_DEVICE)
        print("Image caches were created")

        # Create log & weights directories
        log_dir, weights_dir = _prepare_training_directories(model_file_path)

        # Train 10 instances of the model
        for i in range(1, AMOUNT_OF_MODELS + 1):
            print(f"\n{'='*50}")
            print(f"Starting training for model instance {i}/{AMOUNT_OF_MODELS}")
            print(f"{'='*50}")

            # Create a new model instance
            model = ModelClass().to(TRAINING_DEVICE)

            # Create log file
            log_file_path = _create_log_file(log_dir, i)

            # Train this instance
            train_model(model, train_cache, test_cache, val_cache, log_file_path)

            # Save the model
            _save_model_weights(model, weights_dir, i)

            print(f"Completed training for model instance {i}/{AMOUNT_OF_MODELS}\n")

        print("All training completed!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
