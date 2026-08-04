from ultralytics import YOLO


def train_yolo_model(
    model_path: str = "src/models/yolo26n.pt",  # Path to the YOLO model, weights, or checkpoint
    data_path: str = "src/dataset/data.yaml",
    epochs: int = 2,
    batch: int = 4,
    imgsz: int = 640,
    workers: int = 4,
    optimizer: str = "auto",
    mosaic=0.0,
    cache=False,
    device="cpu",


) -> None:
    """Train a YOLO model with the provided dataset and configuration.

    Args:
        model_path: Path to the YOLO model, weights, or checkpoint.
        data_path: Path to the dataset YAML file.
        epochs: Number of training epochs.
        batch: Batch size for training.
        imgsz: Input image size.
        workers: Number of worker processes for data loading.
        optimizer: Optimization algorithm.
        mosaic: Mosaic augmentation probability.
        cache: Whether to cache images.
        plot: Whether to plot training results.
    """
    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        optimizer=optimizer,
        mosaic=mosaic,
        cache=cache,
        device=device,

    )

if __name__ == "__main__":
    # Example usage
    train_yolo_model()          