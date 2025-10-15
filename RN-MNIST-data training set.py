import numpy as np
import pickle
import os

class Perceptron:
    def __init__(self, input_size=784, output_size=10, learning_rate=0.01):
        """
        Initialize the perceptron with random weights and zero bias.
        
        Args:
            input_size: Number of input features (784 for MNIST)
            output_size: Number of output classes (10 for digits 0-9)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.initial_lr = learning_rate
        
        # Xavier initialization (better for softmax)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.b = np.zeros((output_size,))
        
        # Momentum parameters
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        self.beta = 0.9
    
    def softmax(self, z):
        """
        Apply softmax activation function with numerical stability.
        
        Args:
            z: Weighted sum (m, 10)
        
        Returns:
            Probabilities after softmax (m, 10)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation: compute weighted sum and apply softmax.
        
        Args:
            X: Input matrix of shape (m, 784)
        
        Returns:
            Output probabilities of shape (m, 10)
        """
        # z = W · x + b
        z = np.dot(X, self.W) + self.b
        
        # Apply softmax activation
        y = self.softmax(z)
        
        return y
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Calculate cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities (m, 10)
            y_true: One-hot encoded true labels (m, 10)
        
        Returns:
            Average loss across batch
        """
        m = y_true.shape[0]
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # CrossEntropy = -sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        
        return loss
    
    def backward_propagation(self, X, y_pred, y_true):
        """
        Perform backward propagation and update weights and bias with momentum.
        
        Args:
            X: Input matrix (m, 784)
            y_pred: Predicted probabilities (m, 10)
            y_true: One-hot encoded true labels (m, 10)
        """
        m = X.shape[0]
        
        # Calculate gradient: (Target - y)
        gradient = y_true - y_pred  # (m, 10)
        
        # Calculate weight and bias gradients
        dW = np.dot(X.T, gradient) / m
        db = np.sum(gradient, axis=0) / m
        
        # Update velocities with momentum
        self.v_W = self.beta * self.v_W + (1 - self.beta) * dW
        self.v_b = self.beta * self.v_b + (1 - self.beta) * db
        
        # Update weights with momentum
        self.W += self.learning_rate * self.v_W
        self.b += self.learning_rate * self.v_b
    
    def one_hot_encode(self, y, num_classes=10):
        """
        Convert class labels to one-hot encoding.
        
        Args:
            y: Class labels (m,)
            num_classes: Number of classes
        
        Returns:
            One-hot encoded matrix (m, num_classes)
        """
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y.astype(int)] = 1
        return one_hot
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        """
        Train the perceptron using mini-batch gradient descent with learning rate decay.
        
        Args:
            X_train: Training input (m, 784)
            y_train: Training labels (m,)
            X_val: Validation input (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
        """
        m = X_train.shape[0]
        y_train_encoded = self.one_hot_encode(y_train)
        
        best_val_acc = 0
        patience_counter = 0
        patience = 25  # Increased patience
        
        for epoch in range(epochs):
            # Gentle learning rate decay
            self.learning_rate = self.initial_lr * (0.95 ** (epoch // 10))
            
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward propagation
                y_pred = self.forward_propagation(X_batch)
                
                # Backward propagation
                self.backward_propagation(X_batch, y_pred, y_batch)
            
            # Calculate training loss and accuracy
            y_pred_train = self.forward_propagation(X_train)
            train_loss = self.cross_entropy_loss(y_pred_train, y_train_encoded)
            train_acc = self.accuracy(X_train, y_train)
            
            # Calculate validation accuracy if provided
            if X_val is not None and y_val is not None:
                val_acc = self.accuracy(X_val, y_val)
                print(f"Epoch {epoch+1}/{epochs}, LR: {self.learning_rate:.4f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best weights
                    self.best_W = self.W.copy()
                    self.best_b = self.b.copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc:.4f}")
                    # Restore best weights
                    self.W = self.best_W
                    self.b = self.best_b
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input matrix (m, 784)
        
        Returns:
            Predicted class labels (m,)
        """
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y):
        """
        Calculate accuracy on input data.
        
        Args:
            X: Input matrix (m, 784)
            y: True labels (m,)
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to input data.
    Only adds squared features to keep dimensionality manageable.
    
    Args:
        X: Input matrix (m, 784)
        degree: Polynomial degree (only 2 supported for efficiency)
    
    Returns:
        Enhanced feature matrix
    """
    if degree == 2:
        # Add squared features for non-zero pixels (to avoid too many features)
        X_squared = X ** 2
        # Concatenate original and squared features
        return np.concatenate([X, X_squared], axis=1)
    return X


# ============================================================================
# MAIN: Load data and train perceptron
# ============================================================================

if __name__ == "__main__":
    # Load data from pickle files
    data_path = '/kaggle/input/fii-nn-2025-homework-2'
    
    with open(os.path.join(data_path, 'extended_mnist_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'extended_mnist_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    # Data structure: tuple of (image, label) pairs
    X_train = np.array([item[0] for item in train_data], dtype=np.float32)
    y_train = np.array([item[1] for item in train_data], dtype=np.int32)
    
    X_test = np.array([item[0] for item in test_data], dtype=np.float32)
    
    print(f"Training set shape (before flatten): {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test set shape (before flatten): {X_test.shape}")
    
    # Flatten images from (n, 28, 28) to (n, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Training set shape (after flatten): {X_train.shape}")
    print(f"Test set shape (after flatten): {X_test.shape}")
    
    # Normalize features to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Add polynomial features for better decision boundaries
    print("\nAdding polynomial features...")
    X_train = add_polynomial_features(X_train, degree=2)
    X_test = add_polynomial_features(X_test, degree=2)
    print(f"Enhanced training set shape: {X_train.shape}")
    print(f"Enhanced test set shape: {X_test.shape}")
    
    # Split into training and validation sets (92/8 split)
    n_train = int(0.92 * X_train.shape[0])
    np.random.seed(42)
    indices = np.random.permutation(X_train.shape[0])
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"\nTraining split shape: {X_train_split.shape}")
    print(f"Validation split shape: {X_val.shape}")
    
    print("\n=== Phase 1: Training with validation ===")
    perceptron = Perceptron(input_size=X_train.shape[1], output_size=10, learning_rate=0.3)
    
    perceptron.train(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        epochs=150,
        batch_size=128
    )
    
    val_accuracy = perceptron.accuracy(X_val, y_val)
    print(f"\nPhase 1 Validation Accuracy: {val_accuracy:.4f}")
    
    # Phase 2: Fine-tune on ALL training data
    print("\n=== Phase 2: Fine-tuning on full dataset ===")
    perceptron_final = Perceptron(input_size=X_train.shape[1], output_size=10, learning_rate=0.15)
    
    # Initialize with best weights from phase 1
    perceptron_final.W = perceptron.best_W.copy()
    perceptron_final.b = perceptron.best_b.copy()
    
    # Fine-tune on all 60k samples
    perceptron_final.train(
        X_train, y_train,
        X_val=None, y_val=None,
        epochs=40,
        batch_size=128
    )
    
    # Make predictions on test set
    y_test_pred = perceptron_final.predict(X_test)
    
    print(f"Test predictions shape: {y_test_pred.shape}")
    print(f"Unique predictions: {np.unique(y_test_pred)}")
    
    # Save predictions to CSV for submission
    import pandas as pd
    
    submission_df = pd.DataFrame({
        'ID': np.arange(len(y_test_pred)),
        'target': y_test_pred
    })
    
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    
    print("\nSubmission saved to /kaggle/working/submission.csv")
    print(submission_df.head())
