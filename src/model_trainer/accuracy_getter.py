import os
import sys
import torch
import torch.nn as nn

# Add src to path to import from database_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from image_cache import ImageCache

GREEN_COLOR_ESCAPE = "\033[32m"
RESET_COLOR_ESCAPE = "\033[0m"

def _evaluate_model(model: nn.Module, image_cache: ImageCache, batch_size: int, device: torch.device) -> float:
    """
    Evaluate the model on a database and return (accuracy, average-loss).

    Args:
        model: The PyTorch model to evaluate
        image_cache: ImageCache instance containing pre-loaded images
        batch_size: The maximum batch size to evaluate at parallel
        device: The device to make the evaluation on

    Returns:
        float: Average accuracy (range 0 to 1)
    """
    total_correct = 0
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
            output_classes = torch.argmax(outputs, 1)

            total_correct += torch.sum(batch_labels == output_classes)
            total_loss += loss.item() * len(batch_images)

    return total_correct / len(image_cache), total_loss / len(image_cache)

def evaluate_and_log(model: nn.Module,
                      train_cache: ImageCache,
                      test_cache: ImageCache,
                      val_cache: ImageCache,
                      batch_size: int,
                      device: torch.device,
                      epoch: int | str = "",
                      loss_log_file: str | None = None,
                      accuracy_log_file: str | None = None) -> None:
    """
    Evaluate model and log results.

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
    train_accuracy, train_loss = _evaluate_model(model, train_cache, batch_size, device)
    test_accuracy, test_loss = _evaluate_model(model, test_cache, batch_size, device)
    val_accuracy, val_loss = _evaluate_model(model, val_cache, batch_size, device)

    print(f"{GREEN_COLOR_ESCAPE}Epoch [{epoch}]:" \
        f"\n -- Train Loss: {train_loss:.6f}" \
        f"\n -- Test Loss: {test_loss:.6f}, " \
        f"\n -- Val Loss: {val_loss:.6f}")

    log_content = f"train-accuracy: {100 * train_accuracy:.6f}%\n" \
                f"test-accuracy:  {100 * test_accuracy:.6f}%\n" \
                f"val-accuracy:   {100 * val_accuracy:.6f}%"
    print(log_content, RESET_COLOR_ESCAPE, sep="")


    # Log to CSV
    if (loss_log_file):
        with open(loss_log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{test_loss:.6f},{val_loss:.6f}\n")

    # Log to text file
    if (accuracy_log_file):
        with open(accuracy_log_file, 'w') as f:
            f.write(log_content)
