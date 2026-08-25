import os
import sys
import torch
import torch.nn as nn

# Add src to path to import from database_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from image_cache import ImageCache

def _evaluate_model_accuracy(model: nn.Module, image_cache: ImageCache, batch_size: int, device: torch.device) -> float:
    """
    Evaluate the model on a database and return average loss.

    Args:
        model: The PyTorch model to evaluate
        image_cache: ImageCache instance containing pre-loaded images
        batch_size: The maximum batch size to evaluate at parallel
        device: The device to make the evaluation on

    Returns:
        float: Average accuracy (range 0 to 1)
    """
    correct = 0

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = min(batch_size, len(image_cache))
        num_batches = (len(image_cache) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_cache))

            # Get batch from cache
            batch_images, batch_labels = image_cache[start_idx:end_idx]

            # Move to device, convert type and normalize
            batch_images = batch_images.to(device).to(model.input_dtype) / 255
            # Move to device
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            output_classes = torch.argmax(outputs, 1)

            correct += torch.sum(batch_labels == output_classes)

    return correct / len(image_cache)

def evaluate_accuracy_and_log(model: nn.Module,
                      train_cache: ImageCache,
                      test_cache: ImageCache,
                      val_cache: ImageCache,
                      batch_size: int,
                      device: torch.device,
                      log_file: str) -> None:
    """
    Evaluate model accuracy and log results.

    Args:
        model: The PyTorch model to evaluate
        train_cache: ImageCache containing training images
        test_cache: ImageCache containing test images
        val_cache: ImageCache containing validation images
        batch_size: The maximum batch size to evaluate at parallel
        device: The device to make the evaluation on
        log_file: Path to log file
    """
    model.eval()
    train_accuracy = _evaluate_model_accuracy(model, train_cache, batch_size, device) * 100
    test_accuracy = _evaluate_model_accuracy(model, test_cache, batch_size, device) * 100
    val_accuracy = _evaluate_model_accuracy(model, val_cache, batch_size, device) * 100

    # Log to CSV
    with open(log_file, 'w') as f:
        f.write(f"train-accuracy: {train_accuracy:.6f}%\n" \
                f"test-accuracy:  {test_accuracy:.6f}%\n" \
                f"val_accuracy:   {val_accuracy:.6f}%\n")

    print(f"train-accuracy: {train_accuracy:.6f}%\n" \
          f"test-accuracy:  {test_accuracy:.6f}%\n" \
          f"val_accuracy:   {val_accuracy:.6f}%\n")

def _evaluate_model_loss(model: nn.Module, image_cache: ImageCache, batch_size: int, device: torch.device) -> float:
    """
    Evaluate the model on a database and return average loss.

    Args:
        model: The PyTorch model to evaluate
        image_cache: ImageCache instance containing pre-loaded images
        batch_size: The maximum batch size to evaluate at parallel
        device: The device to make the evaluation on

    Returns:
        float: Average loss over the database
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = min(batch_size, len(image_cache))
        num_batches = (len(image_cache) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_cache))

            # Get batch from cache
            batch_images, batch_labels = image_cache[start_idx:end_idx]

            # Move to device, convert type and normalize
            batch_images = batch_images.to(device).to(model.input_dtype) / 255
            # Move to device
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item() * len(batch_images)

    return total_loss / len(image_cache)

def evaluate_loss_and_log(model: nn.Module,
                      train_cache: ImageCache,
                      test_cache: ImageCache,
                      val_cache: ImageCache,
                      batch_size: int,
                      device: torch.device,
                      epoch: int,
                      log_file: str) -> None:
    """
    Evaluate model and log results.

    Args:
        model: The PyTorch model to evaluate
        train_cache: ImageCache containing training images
        test_cache: ImageCache containing test images
        val_cache: ImageCache containing validation images
        batch_size: The maximum batch size to evaluate at parallel
        device: The device to make the evaluation on
        epoch: Current epoch number
        log_file: Path to log file
    """
    model.eval()
    avg_train_loss = _evaluate_model_loss(model, train_cache, batch_size, device)
    avg_test_loss = _evaluate_model_loss(model, test_cache, batch_size, device)
    avg_val_loss = _evaluate_model_loss(model, val_cache, batch_size, device)

    # Log to CSV
    with open(log_file, 'a') as f:
        f.write(f"{epoch},{avg_train_loss:.6f},{avg_test_loss:.6f},{avg_val_loss:.6f}\n")

    print(f"Epoch [{epoch}], "
        f"Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, "
        f"Val Loss: {avg_val_loss:.6f}")
