import numpy as np
from torchvision.datasets import MNIST
import time


# --- 1. Încărcarea și pregătirea datelor ---

def download_mnist(is_train: bool):
    """
    Descarcă setul de date MNIST folosind torchvision.
    """

    dataset = MNIST(root='./data',
                    train=is_train,
                    transform=lambda x: np.array(x, dtype=np.float32).flatten(),
                    download=True)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return np.array(mnist_data), np.array(mnist_labels)


def preprocess_data(x, y):
    """
    Normalizează imaginile și convertește etichetele în format one-hot.

    """

    x_normalized = x / 255.0

    # One-hot encoding pentru etichete
    num_classes = 10
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    return x_normalized, y_one_hot


# --- 2. Definirea funcțiilor rețelei ---

def softmax(z):
    """
    Calculează funcția softmax pentru un set de intrări.

    """
    # Stabilitate numerică: scădem maximul pentru a evita overflow
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_pass(X, W, b):
    """
    Realizează pasul de forward propagation.

    """
    # Calculează suma ponderată
    z = X @ W + b
    # Aplică funcția de activare softmax [cite: 42]
    y_pred = softmax(z)
    return y_pred


def calculate_accuracy(y_pred, y_true):
    """
    Calculează acuratețea predicțiilor.
    """
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(predicted_classes == true_classes)


# --- 3. Scriptul principal de antrenare ---

if __name__ == "__main__":
    # Încărcare date
    print("Se descarcă setul de date MNIST...")
    train_X_raw, train_Y_raw = download_mnist(True)
    test_X_raw, test_Y_raw = download_mnist(False)

    # Preprocesare date
    train_X, train_Y = preprocess_data(train_X_raw, train_Y_raw)
    test_X, test_Y = preprocess_data(test_X_raw, test_Y_raw)

    print(f"Date de antrenament: {train_X.shape[0]} eșantioane")
    print(f"Date de test: {test_X.shape[0]} eșantioane")

    # Inițializare parametri
    # 784 intrări (28x28 pixeli), 10 ieșiri (cifrele 0-9)
    num_features = 784
    num_classes = 10

    # Inițializare aleatorie pentru ponderi și prag
    np.random.seed(42)  # Pentru reproductibilitate
    W = np.random.randn(num_features, num_classes) * 0.01  #
    b = np.zeros(num_classes)

    # Hiperparametri
    learning_rate = 0.1  # [cite: 59]
    epochs = 100  # Alegem un număr în intervalul recomandat 50-500
    batch_size = 100  #

    # --- Evaluare inițială (înainte de antrenare) ---
    print("\n--- Evaluare înainte de antrenare ---")
    initial_preds = forward_pass(test_X, W, b)
    initial_accuracy = calculate_accuracy(initial_preds, test_Y)
    print(f"Acuratețea inițială pe setul de test: {initial_accuracy * 100:.2f}%")  #

    # --- Bucla de antrenare ---
    print("\n--- Începe antrenamentul ---")
    start_time = time.time()

    for epoch in range(epochs):  #
        # Amestecarea datelor la fiecare epocă
        permutation = np.random.permutation(train_X.shape[0])
        train_X_shuffled = train_X[permutation]
        train_Y_shuffled = train_Y[permutation]

        for i in range(0, train_X.shape[0], batch_size):
            # Extragere batch
            X_batch = train_X_shuffled[i:i + batch_size]
            Y_batch = train_Y_shuffled[i:i + batch_size]

            # Forward Pass [cite: 90]
            y_pred = forward_pass(X_batch, W, b)

            # Backward Pass (Calcul gradient)
            # Gradientul este (y_pred - Target), conform derivatei cross-entropy
            gradient = y_pred - Y_batch

            # Calcul gradient pentru ponderi și prag
            # Se folosește X_batch.T @ gradient pentru o implementare eficientă, vectorizată
            grad_W = (X_batch.T @ gradient) / batch_size
            grad_b = np.sum(gradient, axis=0) / batch_size

            # Actualizare ponderi și prag (Gradient Descent)
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b

        # Afișare acuratețe la fiecare 10 epoci pentru monitorizare
        if (epoch + 1) % 10 == 0:
            current_preds = forward_pass(test_X, W, b)
            current_accuracy = calculate_accuracy(current_preds, test_Y)
            print(f"Epoca {epoch + 1}/{epochs}, Acuratețe test: {current_accuracy * 100:.2f}%")

    end_time = time.time()
    print(f"\nAntrenament finalizat în {end_time - start_time:.2f} secunde.")

    # --- Evaluare finală (după antrenare) ---
    print("\n--- Evaluare după antrenare ---")
    final_preds = forward_pass(test_X, W, b)
    final_accuracy = calculate_accuracy(final_preds, test_Y)
    print(f"Acuratețea finală pe setul de test: {final_accuracy * 100:.2f}%")  #

    # Verificare cerință de acuratețe
    if final_accuracy >= 0.90:
        print("\n🎉 Felicitări! Acuratețea de cel puțin 90% a fost atinsă.")
    else:
        print("\nModelul nu a atins pragul de 90% acuratețe. Încearcă să ajustezi hiperparametrii.")