import os
import sys
from typing import Type
import torch
import torch.nn as nn

# Add src to path to import from database_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database_reader.bird_database import BirdDatabase
from image_cache import ImageCache
from constants import DB_TEST_PATH, DB_TRAIN_PATHS, DB_VALIDATION_PATH, README_PATH, LABEL_NAME_PATH
from model_trainer.model_loader import import_model_class, import_image_size

EVALUATE_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CACHE_DEVICE = torch.device("cpu")
BATCH_SIZE = 2048

MODEL_WEIGHTS_FILE = "/weights/model-1.pt"

def evaluate_model(model: nn.Module, image_cache: ImageCache) -> float:
    """
    Evaluate the model on a database and return average loss.

    Args:
        model: The PyTorch model to evaluate
        image_cache: ImageCache instance containing pre-loaded images

    Returns:
        float: Average accuracy (range 0 to 1)
    """
    correct = 0

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
            batch_images = batch_images.to(EVALUATE_DEVICE).to(model.input_dtype) / 255
            # Move to device
            batch_labels = batch_labels.to(EVALUATE_DEVICE)

            outputs = model(batch_images)
            output_classes = torch.argmax(outputs, 1)

            correct += torch.sum(batch_labels == output_classes)

    return correct / len(image_cache)

def evaluate_and_log(model: nn.Module,
                      train_cache: ImageCache,
                      test_cache: ImageCache,
                      val_cache: ImageCache,
                      log_file: str) -> None:
    """
    Evaluate model accuracy and log results.

    Args:
        model: The PyTorch model to evaluate
        train_cache: ImageCache containing training images
        test_cache: ImageCache containing test images
        val_cache: ImageCache containing validation images
        log_file: Path to log file
    """
    model.eval()
    train_accuracy = evaluate_model(model, train_cache) * 100
    test_accuracy = evaluate_model(model, test_cache) * 100
    val_accuracy = evaluate_model(model, val_cache) * 100

    # Log to CSV
    with open(log_file, 'w') as f:
        f.write(f"train-accuracy: {train_accuracy:.6f}%\n" \
                f"test-accuracy:  {test_accuracy:.6f}%\n" \
                f"val_accuracy:   {val_accuracy:.6f}%\n")

    print(f"train-accuracy: {train_accuracy:.6f}%\n" \
          f"test-accuracy:  {test_accuracy:.6f}%\n" \
          f"val_accuracy:   {val_accuracy:.6f}%\n")

def load_model_weights(weights_file: str, model_type: Type[nn.Module]) -> None:
    """
    Save model weights to file.

    Args:
        model: The PyTorch model to save
        weights_dir: Directory to save weights
        model_id: ID of this model instance

    Returns:
        Path to the saved weights file
    """
    model = model_type().to(EVALUATE_DEVICE)
    state_dict = torch.load(weights_file)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded to {weights_file}")
    return model

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

        model_dir = os.path.dirname(model_file_path)
        model = load_model_weights(model_dir + MODEL_WEIGHTS_FILE, ModelClass)

        evaluate_and_log(model, train_cache, test_cache, val_cache, model_dir + "accuracy.txt")

        print("All training completed!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
