# Command-line Interface (argparse/click)
import argparse
from etna import Model


def main():
    parser = argparse.ArgumentParser(description="Etna CLI")

    parser.add_argument(
        "--file",
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 regularization"
    )

    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction of data used for validation (0 < v < 1)"
    )

    args = parser.parse_args()

    model = Model(args.file, args.target)

    model.train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        validation_split=args.validation_split,
    )


if __name__ == "__main__":
    main()