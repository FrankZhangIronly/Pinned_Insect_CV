import argparse
from ultralytics import YOLO

def train_model(args):
    # Load model with specified .pt file
    model = YOLO(args.model_path)

    # Run training with specified arguments
    model.train(
        data=args.data,
        resume=args.resume,
        device=args.device,
        epochs=args.epochs,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        imgsz=args.imgsz,
        batch=args.batch,
        classes=args.classes,
        project=args.project,
        name=args.name
    )

def validate_model(args):
    # Load model with specified .pt file
    model = YOLO(args.model_path)

    # Run validation for each dataset specified
    for data_path, name in zip(args.data, args.name):
        print(f"Running validation on dataset: {data_path} with name: {name}")
        model.val(data=data_path, imgsz=args.imgsz, batch=args.batch, name=name, classes=args.classes, split="val", workers=args.workers)
        model.val(data=data_path, imgsz=args.imgsz, batch=args.batch, name=f"test_{name}", classes=args.classes, split="test", workers=args.workers)

def main():
    parser = argparse.ArgumentParser(description="Train or Validate YOLO model with custom parameters")

    # Common parameters
    parser.add_argument('--model_path', type=str, required=True, help="Path to the YOLO model .pt file")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--classes', nargs='+', type=int, default=[1, 2, 3], help="List of class IDs, [label, specimen, pin]")
    parser.add_argument('--workers', type=int, default=2, help="Number of data loading workers")

    # Subparsers for train and validate modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose mode: train or validate")

    # Training specific parameters
    train_parser = subparsers.add_parser("train", help="Train the YOLO model")
    train_parser.add_argument('--data', type=str, required=True, help="Path to the dataset YAML file")
    train_parser.add_argument('--resume', type=bool, default=False, help="Resume training from last checkpoint")
    train_parser.add_argument('--device', type=int, default=0, help="GPU device ID")
    train_parser.add_argument('--epochs', type=int, default=300, help="Number of epochs")
    train_parser.add_argument('--close_mosaic', type=int, default=30, help="Epochs to close mosaic augmentation")
    train_parser.add_argument('--project', type=str, default="trains/", help="Project path for saving results")
    train_parser.add_argument('--name', type=str, default="mask_hybrid_mosaic9_v8m_x640", help="Training run name")

    # Validation specific parameters
    validate_parser = subparsers.add_parser("validate", help="Validate the YOLO model")
    validate_parser.add_argument('--data', nargs='+', type=str, required=True, help="List of dataset YAML files for validation (space-separated)")
    validate_parser.add_argument('--name', nargs='+', type=str, required=True, help="List of names for each validation run (space-separated, one for each dataset)")

    args = parser.parse_args()

    # Execute training or validation based on mode
    if args.mode == "train":
        train_model(args)
    elif args.mode == "validate":
        validate_model(args)

if __name__ == "__main__":
    main()
