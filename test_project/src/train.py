#!/usr/bin/env python3
"""
Sample training script for ETNA project
"""
import etna

def main():
    # Load your data here
    X_train = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y_train = [0, 1, 1, 0]
    
    # Train model
    etna.train(X_train, y_train, epochs=100, lr=0.01)
    print("Training complete!")

if __name__ == "__main__":
    main()
