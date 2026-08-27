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
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

# Add src to path to import from database_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database_reader.bird_database import BirdDatabase
from constants import DB_TEST_PATH, DB_TRAIN_PATHS, DB_VALIDATION_PATH, README_PATH, LABEL_NAME_PATH
from image_cache import ImageCache
from model_loader import import_model_class, import_image_size
from accuracy_getter import evaluate_accuracy_and_log, evaluate_loss_and_log

DTYPE = torch.uint8
TRAINING_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CACHE_DEVICE = torch.device("cpu")
BATCH_SIZE = 2048
TRAINING_EPOCHS = 100
LOGGING_INTERVAL = 15
AMOUNT_OF_MODELS = 1
LEARNING_RATE = 0.001

SLEEP_INTERVAL = 10

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

def _create_log_file(log_file: str) -> str:
    """
    Create and initialize the CSV log file.

    Args:
        log_dir: Directory for log files
        model_id: ID of this model instance

    Returns:
        Path to the log file
    """
    
    with open(log_file, 'w') as f:
        f.write("training loops, loss(DB-train), loss(DB-test), loss(DB-validation)\n")
    return log_file

def _train_epoch(model: nn.Module,
                 train_cache: ImageCache,
                 optimizer: optim.Adam,
                 criterion: nn.CrossEntropyLoss,
                 image_transform_filter: v2.Compose | None = None
                 ) -> float:
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

        # Move to device
        batch_images = batch_images.to(TRAINING_DEVICE)
        # Move to device
        batch_labels = batch_labels.to(TRAINING_DEVICE)

        # Run the transform filter
        if image_transform_filter is not None:
            batch_images = image_transform_filter(batch_images)

        # Convert type and normalize
        batch_images = batch_images.to(model.input_dtype) / 255

        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()

    return loss.item()

def save_model_weights(model: nn.Module | optim.Optimizer, weights_file: str) -> None:
    """
    Save model weights to file.

    Args:
        model: The PyTorch model to save
        weights_dir: Directory to save weights
        model_id: ID of this model instance

    Returns:
        Path to the saved weights file
    """
    torch.save(model.state_dict(), weights_file)
    print(f"Model weights saved to {weights_file}")

def load_model_weights(model: nn.Module | optim.Optimizer, weights_file: str) -> None:
    """
    Save model weights to file.

    Args:
        model: The PyTorch model to save
        weights_dir: Directory to save weights
        model_id: ID of this model instance

    Returns:
        Path to the saved weights file
    """
    state_dict = torch.load(weights_file, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from {weights_file}")

def train_model(model: nn.Module,
                optimizer: optim.Optimizer,
                train_cache: ImageCache,
                test_cache: ImageCache,
                val_cache: ImageCache,
                log_file: str,
                image_transform_filter: v2.Compose | None = None
                ):
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

    # Training loop
    for epoch in range(1, TRAINING_EPOCHS + 1):
        if SLEEP_INTERVAL:
            time.sleep(SLEEP_INTERVAL)
        loss = _train_epoch(model, train_cache, optimizer, criterion, image_transform_filter)
        print(f"loss = {loss:.6f}")

        # Evaluate and log every several epochs
        if epoch % LOGGING_INTERVAL == 0 or epoch == TRAINING_DEVICE or epoch == 1:
            evaluate_loss_and_log(model, train_cache, test_cache, val_cache, BATCH_SIZE, TRAINING_DEVICE, epoch, log_file)

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
        image_size = import_image_size(model_file_path)
        print(f"Successfully imported model class: {ModelClass.__name__}")

        # Make the image 'transforms.v2.Compose' filter
        image_transform_filter = v2.Compose([
            v2.RandomResizedCrop(
                size=image_size,
                scale=(0.8, 1),
                ratio=(3/4, 4/3)
            ),
            v2.RandomErasing(
                p=0.5,
                scale=(0.01, 0.05), # (1%, 5%)
                ratio=(3/4, 4/3)
            ),
            v2.RandomErasing(
                p=0.5,
                scale=(0.01, 0.05), # (1%, 5%)
                ratio=(3/4, 4/3)
            ),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(
                brightness=(0.7, 1.3),
                contrast=(0.7, 1.3),
                saturation=(0.7, 1.3),
                hue=(-0.05, 0.05)
            )
        ])

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

            # Define the save paths
            weights_file = os.path.join(weights_dir, f'model-{i}.pt')
            optimizer_file = os.path.join(weights_dir, f'model-{i}-optimizer.pt')
            accuracy_log_file = os.path.join(log_dir, f'model-{i}-accuracy.txt')
            log_file = os.path.join(log_dir, f'train-{i}.csv')

            # Create a new model instance
            model = ModelClass().to(TRAINING_DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            if os.path.exists(weights_file):
                load_model_weights(model, weights_file)
                load_model_weights(optimizer, weights_file)
            else:
                # Create log file
                _create_log_file(log_file)

            # Train this instance
            train_model(model, optimizer, train_cache, test_cache, val_cache, log_file, image_transform_filter)

            # Save the model
            save_model_weights(model, weights_file)
            save_model_weights(optimizer, weights_file)
            evaluate_accuracy_and_log(model, train_cache, test_cache, val_cache, BATCH_SIZE, TRAINING_DEVICE, accuracy_log_file)

            print(f"Completed training for model instance {i}/{AMOUNT_OF_MODELS}\n")

        print("All training completed!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
