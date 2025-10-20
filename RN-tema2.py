import numpy as np
import pickle
import os
from typing import Optional, Tuple
import json


class ImprovedPerceptron:
    
    def __init__(
        self, 
        input_size: int = 784, 
        output_size: int = 10, 
        learning_rate: float = 0.01,
        l2_lambda: float = 0.0,
        use_momentum: bool = True,
        momentum_beta: float = 0.9
    ):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.l2_lambda = l2_lambda
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta
        
        # Xavier/Glorot initialization for weights
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.b = np.zeros((output_size,))
        
        # Momentum velocities
        if use_momentum:
            self.v_W = np.zeros_like(self.W)
            self.v_b = np.zeros_like(self.b)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_weights = None
        self.best_bias = None
        self.best_val_accuracy = 0.0
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.W) + self.b
        y = self.softmax(z)
        return y
    
    def cross_entropy_loss(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        include_regularization: bool = True
    ) -> float:
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        ce_loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        
        if include_regularization and self.l2_lambda > 0:
            l2_penalty = (self.l2_lambda / (2 * m)) * np.sum(self.W ** 2)
            return ce_loss + l2_penalty
        
        return ce_loss
    
    def backward_propagation(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        m = X.shape[0]
        gradient = y_pred - y_true
        
        # Compute gradients
        dW = np.dot(X.T, gradient) / m
        if self.l2_lambda > 0:
            dW += (self.l2_lambda / m) * self.W
        db = np.sum(gradient, axis=0) / m
        
        # Update with momentum
        if self.use_momentum:
            self.v_W = self.momentum_beta * self.v_W + (1 - self.momentum_beta) * dW
            self.v_b = self.momentum_beta * self.v_b + (1 - self.momentum_beta) * db
            self.W -= self.learning_rate * self.v_W
            self.b -= self.learning_rate * self.v_b
        else:
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
        early_stopping_patience: int = 15,
        lr_schedule: str = 'step',  # 'step', 'exponential', or 'none'
        lr_decay_rate: float = 0.95,
        lr_decay_epochs: int = 10,
        verbose: bool = True
    ) -> dict:
        m = X_train.shape[0]
        y_train_encoded = self.one_hot_encode(y_train)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Learning rate scheduling
            if lr_schedule == 'step':
                self.learning_rate = self.initial_lr * (lr_decay_rate ** (epoch // lr_decay_epochs))
            elif lr_schedule == 'exponential':
                self.learning_rate = self.initial_lr * (lr_decay_rate ** epoch)
            
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_pred, y_batch)
            
            # Calculate metrics
            y_pred_train = self.forward_propagation(X_train)
            train_loss = self.cross_entropy_loss(y_pred_train, y_train_encoded)
            train_acc = self.accuracy(X_train, y_train)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['learning_rate'].append(self.learning_rate)
            
            # Validation
            if X_val is not None and y_val is not None:
                y_val_encoded = self.one_hot_encode(y_val)
                y_pred_val = self.forward_propagation(X_val)
                val_loss = self.cross_entropy_loss(y_pred_val, y_val_encoded)
                val_acc = self.accuracy(X_val, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1:3d}/{epochs} - "
                          f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                          f"LR: {self.learning_rate:.5f}")
                
                # Early stopping
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_weights = self.W.copy()
                    self.best_bias = self.b.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n✓ Early stopping at epoch {epoch+1}")
                        print(f"✓ Best validation accuracy: {self.best_val_accuracy:.4f}")
                    self.W = self.best_weights
                    self.b = self.best_bias
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
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
    except FileNotFoundError:
        # Try local paths
        try:
            with open('extended_mnist_train.pkl', 'rb') as f:
                train_data = pickle.load(f)
            with open('extended_mnist_test.pkl', 'rb') as f:
                test_data = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data files not found. Please check the path.") from e
    
    X_train = np.array([item[0] for item in train_data], dtype=np.float32)
    y_train = np.array([item[1] for item in train_data], dtype=np.int32)
    X_test = np.array([item[0] for item in test_data], dtype=np.float32)
    
    return X_train, y_train, X_test


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return X_train, X_test


def train_val_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_ratio: float = 0.15,
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
# MAIN
# ============================================================================

if __name__ == "__main__":
    # OPTIMIZED CONFIGURATION FOR >92.5%
    DATA_PATH = '/kaggle/input/fii-nn-2025-homework-2'
    LEARNING_RATE = 0.3  # Increased from 0.1
    L2_LAMBDA = 0.0001  # Reduced regularization
    EPOCHS = 150  # More epochs
    BATCH_SIZE = 128  # Larger batches
    EARLY_STOPPING_PATIENCE = 20  # More patience
    LR_SCHEDULE = 'step'
    LR_DECAY_RATE = 0.95
    LR_DECAY_EPOCHS = 10
    VAL_RATIO = 0.15  # Less validation, more training
    USE_MOMENTUM = True
    MOMENTUM_BETA = 0.9
    RANDOM_SEED = 42
    
    print("=" * 80)
    print("IMPROVED PERCEPTRON - TARGET: >92.5% ACCURACY")
    print("=" * 80)
    
    np.random.seed(RANDOM_SEED)
    
    # Load and preprocess
    print("\n[1/6] Loading data...")
    X_train, y_train, X_test = load_data(DATA_PATH)
    print(f"✓ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    print("\n[2/6] Preprocessing...")
    X_train, X_test = preprocess_data(X_train, X_test)
    print(f"✓ Preprocessed: Train {X_train.shape}, Test {X_test.shape}")
    
    print("\n[3/6] Train-validation split...")
    X_train_split, y_train_split, X_val, y_val = train_val_split(
        X_train, y_train, val_ratio=VAL_RATIO, random_seed=RANDOM_SEED
    )
    print(f"✓ Train: {X_train_split.shape}, Val: {X_val.shape}")
    
    print("\n[4/6] Training Phase 1 (with validation)...")
    print(f"Config: LR={LEARNING_RATE}, L2={L2_LAMBDA}, Batch={BATCH_SIZE}, Momentum={USE_MOMENTUM}")
    print()
    
    perceptron = ImprovedPerceptron(
        input_size=X_train_split.shape[1],
        output_size=10,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
        use_momentum=USE_MOMENTUM,
        momentum_beta=MOMENTUM_BETA
    )
    
    history = perceptron.train(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        lr_schedule=LR_SCHEDULE,
        lr_decay_rate=LR_DECAY_RATE,
        lr_decay_epochs=LR_DECAY_EPOCHS,
        verbose=True
    )
    
    val_acc_phase1 = perceptron.best_val_accuracy
    print(f"\n✓ Phase 1 Best Val Accuracy: {val_acc_phase1:.4f}")
    
    # Phase 2: Fine-tune on ALL data
    print("\n[5/6] Training Phase 2 (full dataset fine-tuning)...")
    
    perceptron_final = ImprovedPerceptron(
        input_size=X_train.shape[1],
        output_size=10,
        learning_rate=0.1,  # Lower LR for fine-tuning
        l2_lambda=L2_LAMBDA,
        use_momentum=True,
        momentum_beta=0.9
    )
    
    # Transfer weights
    perceptron_final.W = perceptron.best_weights.copy()
    perceptron_final.b = perceptron.best_bias.copy()
    perceptron_final.v_W = np.zeros_like(perceptron_final.W)
    perceptron_final.v_b = np.zeros_like(perceptron_final.b)
    
    history_final = perceptron_final.train(
        X_train, y_train,
        X_val=None, y_val=None,
        epochs=30,
        batch_size=BATCH_SIZE,
        lr_schedule='exponential',
        lr_decay_rate=0.98,
        verbose=True
    )
    
    print("\n[6/6] Generating predictions...")
    y_test_pred = perceptron_final.predict(X_test)
    
    # Save results
    import pandas as pd
    
    submission_df = pd.DataFrame({
        'ID': np.arange(len(y_test_pred)),
        'target': y_test_pred
    })
    
    output_path = '/kaggle/working/submission.csv' if os.path.exists('/kaggle/working') else 'submission.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Phase 1 Best Validation Accuracy: {val_acc_phase1:.4f}")
    print(f"Expected Test Accuracy: ~{val_acc_phase1 + 0.005:.4f} (with full data fine-tuning)")
    print("=" * 80)
