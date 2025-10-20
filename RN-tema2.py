import numpy as np
import pickle
import os

class Perceptron:
    def __init__(self, input_size=784, output_size=10, learning_rate=0.01):
        
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier initialization for weights
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros((output_size,))
    
    def softmax(self, z):
        
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
       
        # z = W · x + b
        z = np.dot(X, self.W) + self.b
        
        # Apply softmax activation
        y = self.softmax(z)
        
        return y
    
    def cross_entropy_loss(self, y_pred, y_true):
       
        m = y_true.shape[0]
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # CrossEntropy = -sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        
        return loss
    
    def backward_propagation(self, X, y_pred, y_true):
        
        m = X.shape[0]
        
        # Calculate gradient: (Target - y)
        gradient = y_true - y_pred  # (m, 10)
        
        # Update weights: W = W + μ × (Target - y) × x^T
        dW = np.dot(X.T, gradient) / m
        self.W += self.learning_rate * dW
        
        # Update bias: b = b + μ × (Target - y)
        db = np.sum(gradient, axis=0) / m
        self.b += self.learning_rate * db
    
    def one_hot_encode(self, y, num_classes=10):
        
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y.astype(int)] = 1
        return one_hot
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
       
        m = X_train.shape[0]
        y_train_encoded = self.one_hot_encode(y_train)
        
        for epoch in range(epochs):
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
            
            # Calculate training loss
            y_pred_train = self.forward_propagation(X_train)
            train_loss = self.cross_entropy_loss(y_pred_train, y_train_encoded)
            
            # Calculate validation accuracy if provided
            if X_val is not None and y_val is not None:
                val_acc = self.accuracy(X_val, y_val)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    def predict(self, X):
       
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y):
        
        predictions = self.predict(X)
        return np.mean(predictions == y)


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
    # Extract images and labels from the tuple of tuples
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
    
    # Split into training and validation sets (80/20 split)
    n_train = int(0.8 * X_train.shape[0])
    indices = np.random.permutation(X_train.shape[0])
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"\nTraining split shape: {X_train_split.shape}")
    print(f"Validation split shape: {X_val.shape}")
    
    # Initialize and train perceptron
    perceptron = Perceptron(input_size=X_train.shape[1], output_size=10, learning_rate=0.1)
    
    perceptron.train(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        epochs=25,
        batch_size=32
    )
    
    # Evaluate on validation set
    val_accuracy = perceptron.accuracy(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
    
    # Make predictions on test set
    y_test_pred = perceptron.predict(X_test)
    
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
