import numpy as np
import pickle
import os
import time
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


class MLP:

    def __init__(
            self,
            input_size: int = 784,
            hidden_size: int = 100,
            output_size: int = 10,
            learning_rate: float = 0.1,
            use_dropout: bool = True,
            dropout_rate: float = 0.5,
            use_momentum: bool = True,
            momentum_beta: float = 0.9
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta

        # He initialization for ReLU activation
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.5 / input_size)
        self.b1 = np.zeros((hidden_size,))

        # Xavier initialization for output layer (softmax)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros((output_size,))

        # Momentum velocities
        if use_momentum:
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)

        self.training = True

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def relu(self, z: np.ndarray) -> np.ndarray:

        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:

        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:

        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def dropout_mask(self, shape: Tuple, p: float) -> np.ndarray:

        mask = (np.random.rand(*shape) > p).astype(float)

        mask /= (1 - p)
        return mask

    def forward_propagation(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, dict]:

        cache = {}

        # Layer 1: Input -> Hidden
        cache['X'] = X
        z1 = np.dot(X, self.W1) + self.b1
        cache['z1'] = z1

        # ReLU activation
        a1 = self.relu(z1)
        cache['a1'] = a1

        # Dropout on hidden layer (only during training)
        if self.use_dropout and training:
            dropout_mask = self.dropout_mask(a1.shape, self.dropout_rate)
            a1 = a1 * dropout_mask
            cache['dropout_mask'] = dropout_mask

        # Layer 2: Hidden -> Output
        z2 = np.dot(a1, self.W2) + self.b2
        cache['z2'] = z2
        cache['a1_after_dropout'] = a1

        # Softmax activation
        output = self.softmax(z2)
        cache['output'] = output

        return output, cache

    def backward_propagation(self, y_true: np.ndarray, cache: dict) -> dict:

        batch_size = y_true.shape[0]
        gradients = {}

        X = cache['X']
        z1 = cache['z1']
        a1_after_dropout = cache['a1_after_dropout']
        output = cache['output']

        # Output Layer
        dz2 = output - y_true
        dW2 = np.dot(a1_after_dropout.T, dz2) / batch_size
        gradients['dW2'] = dW2
        db2 = np.sum(dz2, axis=0) / batch_size
        gradients['db2'] = db2

        # Hidden Layer
        da1 = np.dot(dz2, self.W2.T)
        if self.use_dropout and 'dropout_mask' in cache:
            da1 = da1 * cache['dropout_mask']
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = np.dot(X.T, dz1) / batch_size
        gradients['dW1'] = dW1
        db1 = np.sum(dz1, axis=0) / batch_size
        gradients['db1'] = db1

        return gradients

    def update_parameters(self, gradients: dict):

        if self.use_momentum:
            # Momentum update
            self.v_W2 = self.momentum_beta * self.v_W2 + (1 - self.momentum_beta) * gradients['dW2']
            self.v_b2 = self.momentum_beta * self.v_b2 + (1 - self.momentum_beta) * gradients['db2']
            self.v_W1 = self.momentum_beta * self.v_W1 + (1 - self.momentum_beta) * gradients['dW1']
            self.v_b1 = self.momentum_beta * self.v_b1 + (1 - self.momentum_beta) * gradients['db1']

            # Update with momentum
            self.W2 -= self.learning_rate * self.v_W2
            self.b2 -= self.learning_rate * self.v_b2
            self.W1 -= self.learning_rate * self.v_W1
            self.b1 -= self.learning_rate * self.v_b1
        else:
            # Standard gradient descent
            self.W2 -= self.learning_rate * gradients['dW2']
            self.b2 -= self.learning_rate * gradients['db2']
            self.W1 -= self.learning_rate * gradients['dW1']
            self.b1 -= self.learning_rate * gradients['db1']

    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:

        y_pred_torch = torch.from_numpy(y_pred).float()
        y_true_indices = torch.from_numpy(np.argmax(y_true, axis=1)).long()
        loss = F.cross_entropy(y_pred_torch, y_true_indices)
        return loss.item()

    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:

        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y.astype(int)] = 1
        return one_hot

    def train_model(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int = 50,
            batch_size: int = 128,
            verbose: bool = True
    ):

        m = X_train.shape[0]
        y_train_encoded = self.one_hot_encode(y_train)
        y_val_encoded = self.one_hot_encode(y_val)

        best_val_acc = 0
        plateau_counter = 0

        for epoch in range(epochs):
            # Shuffle and batch
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output, cache = self.forward_propagation(X_batch, training=True)
                gradients = self.backward_propagation(y_batch, cache)
                self.update_parameters(gradients)

            # Evaluate
            train_output, _ = self.forward_propagation(X_train, training=False)
            train_loss = self.cross_entropy_loss(train_output, y_train_encoded)
            train_acc = self.accuracy(X_train, y_train)

            val_output, _ = self.forward_propagation(X_val, training=False)
            val_loss = self.cross_entropy_loss(val_output, y_val_encoded)
            val_acc = self.accuracy(X_val, y_val)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            # Learning rate scheduling
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                plateau_counter = 0
            else:
                plateau_counter += 1
                if plateau_counter >= 5:
                    self.learning_rate *= 0.7
                    plateau_counter = 0
                    if verbose:
                        print(f"  → LR decayed to {self.learning_rate:.6f}")

            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {self.learning_rate:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:

        output, _ = self.forward_propagation(X, training=False)
        return np.argmax(output, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:

        predictions = self.predict(X)
        return np.mean(predictions == y)


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(data_path, 'extended_mnist_train.pkl')
    test_path = os.path.join(data_path, 'extended_mnist_test.pkl')

    with open(train_path, 'rb') as f: train_data = pickle.load(f)
    with open(test_path, 'rb') as f: test_data = pickle.load(f)

    X_train = np.array([item[0] for item in train_data], dtype=np.float32)
    y_train = np.array([item[1] for item in train_data], dtype=np.int32)
    X_test = np.array([item[0] for item in test_data], dtype=np.float32)

    return X_train, y_train, X_test


if __name__ == "__main__":
    print("=" * 80)
    print("MLP IMPLEMENTATION - TARGET: >98% ACCURACY IN < 8 MINUTES")
    print("=" * 80)

    start_time = time.time()
    np.random.seed(42)

    DATA_PATH = '/kaggle/input/fii-nn-2025-homework-3'
    LEARNING_RATE = 0.9
    EPOCHS = 80
    BATCH_SIZE = 512
    DROPOUT_RATE = 0.2
    HIDDEN_SIZE = 224
    MOMENTUM_BETA = 0.9
    FINETUNE_EPOCHS = 30

    # 1. Load data
    print("\n[1/5] Loading data...")
    X_train, y_train, X_test = load_data(DATA_PATH)

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # 3. Train/val split
    print("\n[3/5] Train-validation split (90/10)...")
    n_train = int(0.9 * len(X_train))
    indices = np.random.permutation(len(X_train))
    X_train_split, y_train_split = X_train[indices[:n_train]], y_train[indices[:n_train]]
    X_val, y_val = X_train[indices[n_train:]], y_train[indices[n_train:]]

    # 4. Train model
    print("\n[4/5] Training MLP...")
    print(f"Architecture: 784 → {HIDDEN_SIZE} (ReLU + Dropout {DROPOUT_RATE}) → 10 (Softmax)")
    print(f"Config: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")

    mlp = MLP(
        input_size=784, hidden_size=HIDDEN_SIZE, output_size=10,
        learning_rate=LEARNING_RATE, use_dropout=True, dropout_rate=DROPOUT_RATE,
        use_momentum=True, momentum_beta=MOMENTUM_BETA
    )
    mlp.train_model(
        X_train_split, y_train_split, X_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True
    )

    # 5. Evaluate
    val_acc = mlp.accuracy(X_val, y_val)
    print(f"\n[5/5] Validation Accuracy after initial training: {val_acc:.4f}")

    # 6. Fine-tune on full dataset
    print("\n[6/6] Fine-tuning on full dataset...")
    mlp.learning_rate = 0.02
    mlp.use_dropout = False
    mlp.momentum_beta = 0.95
    mlp.v_W1, mlp.v_b1, mlp.v_W2, mlp.v_b2 = [np.zeros_like(p) for p in [mlp.W1, mlp.b1, mlp.W2, mlp.b2]]

    y_train_encoded = mlp.one_hot_encode(y_train)
    for epoch in range(FINETUNE_EPOCHS):
        indices = np.random.permutation(len(X_train))
        X_shuffled, y_shuffled = X_train[indices], y_train_encoded[indices]

        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch, y_batch = X_shuffled[i:i + BATCH_SIZE], y_shuffled[i:i + BATCH_SIZE]
            output, cache = mlp.forward_propagation(X_batch, training=False)
            gradients = mlp.backward_propagation(y_batch, cache)
            mlp.update_parameters(gradients)

        if (epoch + 1) % 10 == 0: mlp.learning_rate *= 0.8
        if (epoch + 1) % 5 == 0:
            train_acc = mlp.accuracy(X_train, y_train)
            print(
                f"Fine-tune Epoch {epoch + 1}/{FINETUNE_EPOCHS} | Train Acc: {train_acc:.4f} | LR: {mlp.learning_rate:.6f}")

    # Generate and save predictions
    y_test_pred = mlp.predict(X_test)
    import pandas as pd

    submission_df = pd.DataFrame({'ID': np.arange(len(y_test_pred)), 'target': y_test_pred})
    output_file = 'submission.csv'
    submission_df.to_csv(output_file, index=False)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE IN {elapsed_time:.2f} SECONDS ({elapsed_time / 60:.2f} MINUTES)")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print("=" * 80)