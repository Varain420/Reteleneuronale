import numpy as np
import pickle
import os
from typing import Optional, Tuple
import json


class Perceptron:
    
    
    def __init__(
        self, 
        input_size: int = 784, 
        output_size: int = 10, 
        learning_rate: float = 0.01,
        l2_lambda: float = 0.0
    ):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.l2_lambda = l2_lambda
        
        # Xavier/Glorot initialization for weights
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.b = np.zeros((output_size,))
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Best model tracking for early stopping
        self.best_weights = None
        self.best_bias = None
        self.best_val_accuracy = 0.0
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        
        # Linear transformation: z = Wx + b
        z = np.dot(X, self.W) + self.b
        
        # Apply softmax activation
        y = self.softmax(z)
        
        return y
    
    def cross_entropy_loss(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        include_regularization: bool = True
    ) -> float:
        
        m = y_true.shape[0]
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy: -sum(y_true * log(y_pred)) / m
        ce_loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        
        # Add L2 regularization if enabled
        if include_regularization and self.l2_lambda > 0:
            l2_penalty = (self.l2_lambda / (2 * m)) * np.sum(self.W ** 2)
            return ce_loss + l2_penalty
        
        return ce_loss
    
    def backward_propagation(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        
        m = X.shape[0]
        
        # CORRECTED: Calculate gradient as (y_pred - y_true) for proper gradient descent
        # This is the derivative of cross-entropy loss with softmax
        gradient = y_pred - y_true  # Shape: (m, output_size)
        
        # Compute weight gradient with L2 regularization
        dW = np.dot(X.T, gradient) / m  # Shape: (input_size, output_size)
        
        # Add L2 regularization gradient
        if self.l2_lambda > 0:
            dW += (self.l2_lambda / m) * self.W
        
        # Compute bias gradient
        db = np.sum(gradient, axis=0) / m  # Shape: (output_size,)
        
        # CORRECTED: Subtract gradients (gradient descent, not ascent)
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
    
    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y.astype(int)] = 1
        return one_hot
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        learning_rate_decay: float = 1.0,
        verbose: bool = True
    ) -> dict:
        
        m = X_train.shape[0]
        y_train_encoded = self.one_hot_encode(y_train)
        
        # For early stopping
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward propagation
                y_pred = self.forward_propagation(X_batch)
                
                # Backward propagation
                self.backward_propagation(X_batch, y_pred, y_batch)
            
            # Calculate metrics on full training set
            y_pred_train = self.forward_propagation(X_train)
            train_loss = self.cross_entropy_loss(y_pred_train, y_train_encoded)
            self.history['train_loss'].append(train_loss)
            
            # Calculate validation metrics if provided
            if X_val is not None and y_val is not None:
                y_val_encoded = self.one_hot_encode(y_val)
                y_pred_val = self.forward_propagation(X_val)
                val_loss = self.cross_entropy_loss(y_pred_val, y_val_encoded)
                val_acc = self.accuracy(X_val, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Accuracy: {val_acc:.4f}, "
                          f"LR: {self.learning_rate:.6f}")
                
                # Early stopping logic
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_weights = self.W.copy()
                    self.best_bias = self.b.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
                    # Restore best weights
                    self.W = self.best_weights
                    self.b = self.best_bias
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
            
            # Learning rate decay
            if learning_rate_decay < 1.0:
                self.learning_rate *= learning_rate_decay
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        
        return self.forward_propagation(X)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def save_model(self, filepath: str) -> None:
        
        model_data = {
            'weights': self.W.tolist(),
            'bias': self.b.tolist(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'l2_lambda': self.l2_lambda,
            'history': self.history
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.W = np.array(model_data['weights'])
        self.b = np.array(model_data['bias'])
        self.input_size = model_data['input_size']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.l2_lambda = model_data['l2_lambda']
        self.history = model_data['history']


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    try:
        with open(os.path.join(data_path, 'extended_mnist_train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        
        with open(os.path.join(data_path, 'extended_mnist_test.pkl'), 'rb') as f:
            test_data = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data files not found at {data_path}. Please check the path.") from e
    
    # Extract images and labels
    X_train = np.array([item[0] for item in train_data], dtype=np.float32)
    y_train = np.array([item[1] for item in train_data], dtype=np.int32)
    X_test = np.array([item[0] for item in test_data], dtype=np.float32)
    
    return X_train, y_train, X_test


def preprocess_data(
    X_train: np.ndarray, 
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    
    # Flatten images from (n, 28, 28) to (n, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Normalize to [0, 1] range
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return X_train, X_test


def train_val_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_total = X.shape[0]
    n_train = int((1 - val_ratio) * n_total)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


# ============================================================================
# MAIN: Load data and train perceptron
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = '/kaggle/input/fii-nn-2025-homework-2'
    LEARNING_RATE = 0.1
    L2_LAMBDA = 0.001  # Small regularization
    EPOCHS = 50
    BATCH_SIZE = 64
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE_DECAY = 0.95
    RANDOM_SEED = 42
    
    print("=" * 70)
    print("PERCEPTRON TRAINING - EXTENDED MNIST CLASSIFICATION")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Load data
    print("\n[1/5] Loading data...")
    try:
        X_train, y_train, X_test = load_data(DATA_PATH)
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Training labels: {y_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        exit(1)
    
    # Preprocess data
    print("\n[2/5] Preprocessing data...")
    X_train, X_test = preprocess_data(X_train, X_test)
    print(f"✓ Training set (flattened): {X_train.shape}")
    print(f"✓ Test set (flattened): {X_test.shape}")
    print(f"✓ Features normalized to [0, 1]")
    
    # Split into training and validation
    print("\n[3/5] Creating train-validation split...")
    X_train_split, y_train_split, X_val, y_val = train_val_split(
        X_train, y_train, val_ratio=0.2, random_seed=RANDOM_SEED
    )
    print(f"✓ Training set: {X_train_split.shape}")
    print(f"✓ Validation set: {X_val.shape}")
    
    # Initialize perceptron
    print("\n[4/5] Initializing and training perceptron...")
    print(f"Configuration:")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - L2 regularization: {L2_LAMBDA}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max epochs: {EPOCHS}")
    print(f"  - Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  - Learning rate decay: {LEARNING_RATE_DECAY}")
    print()
    
    perceptron = Perceptron(
        input_size=X_train_split.shape[1],
        output_size=10,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA
    )
    
    # Train the model
    history = perceptron.train(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        learning_rate_decay=LEARNING_RATE_DECAY,
        verbose=True
    )
    
    # Final evaluation
    print("\n[5/5] Final evaluation...")
    val_accuracy = perceptron.accuracy(X_val, y_val)
    print(f"✓ Final Validation Accuracy: {val_accuracy:.4f}")
    
    # Make predictions on test set
    print("\nGenerating test predictions...")
    y_test_pred = perceptron.predict(X_test)
    print(f"✓ Test predictions shape: {y_test_pred.shape}")
    print(f"✓ Unique predictions: {np.unique(y_test_pred)}")
    
    # Save predictions
    import pandas as pd
    
    submission_df = pd.DataFrame({
        'ID': np.arange(len(y_test_pred)),
        'target': y_test_pred
    })
    
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    print(f"✓ Submission saved to /kaggle/working/submission.csv")
    
    # Save model
    perceptron.save_model('/kaggle/working/perceptron_model.json')
    print(f"✓ Model saved to /kaggle/working/perceptron_model.json")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {perceptron.best_val_accuracy:.4f}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
