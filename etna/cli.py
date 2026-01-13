"""
Command-line Interface for ETNA neural network framework.
"""

import argparse
import sys
import os
from .api import Model


def train_command(args):
    """Train a model from command line."""
    try:
        print(f"[*] Loading data from {args.data}")
        model = Model(
            file_path=args.data,
            target=args.target,
            task_type=args.task_type
        )
        
        model.train(
            epochs=args.epochs,
            lr=args.learning_rate,
            optimizer=args.optimizer
        )
        
        if args.save:
            print(f"[*] Saving model to {args.save}...")
            model.save_model(path=args.save, run_name=args.run_name)
        else:
            print("[*] Model trained successfully!")
            print("[!] Use --save to save the model for later use")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)


def predict_command(args):
    """Make predictions using a trained model."""
    try:
        if not args.model:
            print("[ERROR] --model is required for predictions", file=sys.stderr)
            print("[!] First train a model and save it, then load it for predictions", file=sys.stderr)
            sys.exit(1)
        
        if not args.data:
            print("[ERROR] --data is required for predictions", file=sys.stderr)
            sys.exit(1)
        
        # Load saved model
        print(f"[*] Loading model from {args.model}...")
        print("[!] Note: Predictions require the same data structure as training.")
        print("[!] For best results, use the Python API directly after training.")
        
        model = Model.load(args.model)
        
        print(f"[*] Making predictions on {args.data}...")
        predictions = model.predict(data_path=args.data)
        
        # Output predictions
        if args.output:
            with open(args.output, 'w') as f:
                f.write("index,prediction\n")
                for i, pred in enumerate(predictions):
                    f.write(f"{i},{pred}\n")
            print(f"[*] Predictions saved to {args.output}")
        else:
            print("\n[*] Predictions:")
            for i, pred in enumerate(predictions[:10]):  # Show first 10
                print(f"  Sample {i}: {pred}")
            if len(predictions) > 10:
                print(f"  ... and {len(predictions) - 10} more")
                
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}", file=sys.stderr)
        print("[!] Tip: Predictions work best when using the Python API directly after training.", file=sys.stderr)
        print("[!] The preprocessor state needs to match between training and prediction.", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ETNA - High-Performance Neural Networks with Rust Core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a classification model with Adam optimizer
  etna train data.csv --target species --optimizer adam --epochs 100 --save model.json
  
  # Train a regression model
  etna train housing.csv --target price --task-type regression --optimizer adam
  
  # Make predictions
  etna predict --model model.json --data test.csv --output predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a neural network model')
    train_parser.add_argument('data', help='Path to CSV data file')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--task-type', choices=['classification', 'regression'],
                               help='Task type (auto-detected if not specified)')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs (default: 100)')
    train_parser.add_argument('--learning-rate', '--lr', type=float, default=0.01,
                             help='Learning rate (default: 0.01)')
    train_parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd',
                             help='Optimizer to use: sgd or adam (default: sgd)')
    train_parser.add_argument('--save', help='Path to save the trained model')
    train_parser.add_argument('--run-name', default='ETNA_CLI_Run',
                             help='MLflow run name (default: ETNA_CLI_Run)')
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions using a trained model')
    predict_parser.add_argument('--model', required=True, help='Path to saved model file')
    predict_parser.add_argument('--data', required=True, help='Path to CSV data file for predictions')
    predict_parser.add_argument('--output', help='Output file to save predictions (CSV format)')
    predict_parser.set_defaults(func=predict_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
