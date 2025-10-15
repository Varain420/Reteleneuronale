import numpy as np
import pickle
import os


class MLP:
    def __init__(self, input_size=784, hidden_size=256, output_size=10, learning_rate=0.01):
        """
        Initialize Multi-Layer Perceptron with one hidden layer.

        Args:
            input_size: Number of input features (784 for MNIST)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes (10 for digits 0-9)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initial_lr = learning_rate

        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size,))

        # Xavier initialization for output layer (softmax)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros((output_size,))

        # Momentum parameters
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.beta = 0.9

    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Apply softmax activation function with numerical stability.

        Args:
            z: Weighted sum (m, 10)

        Returns:
            Probabilities after softmax (m, 10)
        """
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.

        Args:
            X: Input matrix of shape (m, 784)

        Returns:
            Output probabilities of shape (m, 10)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

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
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
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

        # Output layer gradients
        dz2 = y_pred - y_true  # Gradient of cross-entropy with softmax
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # Update velocities with momentum
        self.v_W2 = self.beta * self.v_W2 + (1 - self.beta) * dW2
        self.v_b2 = self.beta * self.v_b2 + (1 - self.beta) * db2
        self.v_W1 = self.beta * self.v_W1 + (1 - self.beta) * dW1
        self.v_b1 = self.beta * self.v_b1 + (1 - self.beta) * db1

        # Update weights with momentum
        self.W2 -= self.learning_rate * self.v_W2
        self.b2 -= self.learning_rate * self.v_b2
        self.W1 -= self.learning_rate * self.v_W1
        self.b1 -= self.learning_rate * self.v_b1

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
        Train the MLP using mini-batch gradient descent with learning rate decay.

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
        patience = 15

        for epoch in range(epochs):
            # Learning rate decay
            self.learning_rate = self.initial_lr * (0.95 ** (epoch // 10))

            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]

            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

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
                print(
                    f"Epoch {epoch + 1}/{epochs}, LR: {self.learning_rate:.4f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best weights
                    self.best_W1 = self.W1.copy()
                    self.best_b1 = self.b1.copy()
                    self.best_W2 = self.W2.copy()
                    self.best_b2 = self.b2.copy()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_acc:.4f}")
                    # Restore best weights
                    self.W1 = self.best_W1
                    self.b1 = self.best_b1
                    self.W2 = self.best_W2
                    self.b2 = self.best_b2
                    break
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

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


# ============================================================================
# MAIN: Load data and train MLP
# ============================================================================

if __name__ == "__main__":
    # Detect if running on Kaggle or locally
    import sys

    # Try Kaggle paths first
    kaggle_path = '/kaggle/input/fii-nn-2025-homework-2'
    local_train_path = 'extended_mnist_train.pkl'
    local_test_path = 'extended_mnist_test.pkl'

    # Check if running on Kaggle
    if os.path.exists(kaggle_path):
        print("Running on Kaggle...")
        data_path = kaggle_path
        train_file = os.path.join(data_path, 'extended_mnist_train.pkl')
        test_file = os.path.join(data_path, 'extended_mnist_test.pkl')
    # Check if running locally with files in current directory
    elif os.path.exists(local_train_path) and os.path.exists(local_test_path):
        print("Running locally with files in current directory...")
        train_file = local_train_path
        test_file = local_test_path
    else:
        print("ERROR: Data files not found!")
        print(
            "Please place 'extended_mnist_train.pkl' and 'extended_mnist_test.pkl' in the same directory as this script.")
        sys.exit(1)

    # Load data from pickle files
    print(f"Loading training data from: {train_file}")
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading test data from: {test_file}")
    with open(test_file, 'rb') as f:
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

    # Split into training and validation sets (90/10 split for more training data)
    n_train = int(0.9 * X_train.shape[0])
    np.random.seed(42)  # Fixed seed for reproducibility
    indices = np.random.permutation(X_train.shape[0])

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(f"\nTraining split shape: {X_train_split.shape}")
    print(f"Validation split shape: {X_val.shape}")

    # Initialize and train MLP with optimized hyperparameters
    print("\n=== Phase 1: Training MLP with validation ===")
    mlp = MLP(input_size=X_train.shape[1], hidden_size=256, output_size=10, learning_rate=0.1)

    mlp.train(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        epochs=100,
        batch_size=128
    )

    # Evaluate on validation set
    val_accuracy = mlp.accuracy(X_val, y_val)
    print(f"\nPhase 1 Validation Accuracy: {val_accuracy:.4f}")

    # Phase 2: Fine-tune on ALL training data for final submission
    print("\n=== Phase 2: Fine-tuning on full dataset ===")
    mlp_final = MLP(input_size=X_train.shape[1], hidden_size=256, output_size=10, learning_rate=0.05)

    # Initialize with best weights from phase 1
    mlp_final.W1 = mlp.best_W1.copy()
    mlp_final.b1 = mlp.best_b1.copy()
    mlp_final.W2 = mlp.best_W2.copy()
    mlp_final.b2 = mlp.best_b2.copy()

    # Fine-tune on all 60k samples
    mlp_final.train(
        X_train, y_train,
        X_val=None, y_val=None,
        epochs=30,
        batch_size=128
    )

    # Use the fine-tuned model for predictions
    mlp = mlp_final

    # Make predictions on test set
    y_test_pred = mlp.predict(X_test)

    print(f"Test predictions shape: {y_test_pred.shape}")
    print(f"Unique predictions: {np.unique(y_test_pred)}")

    # Save predictions to CSV for submission
    import pandas as pd

    submission_df = pd.DataFrame({
        'ID': np.arange(len(y_test_pred)),
        'target': y_test_pred
    })

    # Save to current directory (works both locally and on Kaggle)
    output_file = 'submission.csv'
    if os.path.exists('/kaggle/working'):
        output_file = '/kaggle/working/submission.csv'

    submission_df.to_csv(output_file, index=False)

    print(f"\nSubmission saved to {output_file}")
    print(submission_df.head())