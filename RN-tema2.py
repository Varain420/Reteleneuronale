import numpy as np
import time
import pickle
import pandas as pd
from torchvision.datasets import MNIST


# --- PASUL 1: Încărcarea și Pregătirea Datelor (Metoda Robustă) ---

def descarca_mnist(este_antrenament: bool):
    """
    Descarcă setul de date MNIST folosind torchvision și îl convertește în format NumPy.
    """
    print(f"Se descarcă setul de date {'de antrenament' if este_antrenament else 'de test'}...")
    dataset = MNIST(root='./data',
                    train=este_antrenament,
                    transform=lambda img: np.array(img, dtype=np.float32).flatten(),
                    download=True)

    date_imagini = []
    date_etichete = []
    for imagine, eticheta in dataset:
        date_imagini.append(imagine)
        date_etichete.append(eticheta)

    return np.array(date_imagini), np.array(date_etichete)


def preproceseaza_date(x, y):
    """
    Normalizează imaginile și convertește etichetele în format one-hot.
    """
    x_normalizat = x / 255.0

    nr_clase = 10
    y_one_hot = np.zeros((y.shape[0], nr_clase))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    return x_normalizat, y_one_hot


def incarca_date_competitie(cale_fisier_pkl):
    """
    Funcție specializată pentru a încărca fișierul de test al competiției.
    """
    print(f"Se încarcă datele de competiție din '{cale_fisier_pkl}'...")
    with open(cale_fisier_pkl, 'rb') as f:
        data_bruta = pickle.load(f, encoding='latin1')

    imagini = data_bruta[0]
    while isinstance(imagini, tuple):
        imagini = imagini[0]

    if imagini.ndim == 3:
        nr_imagini = imagini.shape[0]
        imagini_vectorizate = imagini.reshape(nr_imagini, -1)
    else:
        imagini_vectorizate = imagini.flatten().reshape(1, -1)

    if imagini_vectorizate.shape[1] != 784:
        raise ValueError("Dimensiunea caracteristicilor nu este 784.")

    return imagini_vectorizate / 255.0


# --- PASUL 2: Definirea Funcțiilor Rețelei Multi-Strat ---

def relu(z):
    """Funcția de activare ReLU."""
    return np.maximum(0, z)


def derivata_relu(z):
    """Derivata funcției ReLU."""
    return z > 0


def softmax(z):
    """Funcția Softmax stabilă numeric."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def propagare_inainte(X, W1, b1, W2, b2):
    """Calculează output-ul rețelei și valorile intermediare."""
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def calculeaza_acuratete(y_pred_one_hot, y_adevarat_one_hot):
    """Calculează acuratețea."""
    predictii = np.argmax(y_pred_one_hot, axis=1)
    etichete_reale = np.argmax(y_adevarat_one_hot, axis=1)
    return np.mean(predictii == etichete_reale)


# --- PASUL 3: Execuția Principală ---
if __name__ == '__main__':
    # --- Încărcarea datelor ---
    X_antrenare_brut, y_antrenare_brut = descarca_mnist(este_antrenament=True)
    X_validare_brut, y_validare_brut = descarca_mnist(este_antrenament=False)

    X_antrenare, Y_antrenare = preproceseaza_date(X_antrenare_brut, y_antrenare_brut)
    X_validare, Y_validare = preproceseaza_date(X_validare_brut, y_validare_brut)

    print(f"\nForma datelor de antrenare: {X_antrenare.shape}")
    print(f"Forma datelor de validare: {X_validare.shape}")

    # --- Inițializarea Ponderilor și Hiperparametrilor pentru MLP ---
    nr_caracteristici = X_antrenare.shape[1]
    nr_neuroni_ascunsi = 128
    nr_clase = Y_antrenare.shape[1]

    W1 = np.random.randn(nr_caracteristici, nr_neuroni_ascunsi) * 0.01
    b1 = np.zeros((1, nr_neuroni_ascunsi))
    W2 = np.random.randn(nr_neuroni_ascunsi, nr_clase) * 0.01
    b2 = np.zeros((1, nr_clase))

    rata_invatare = 0.1
    epoci = 50
    dimensiune_batch = 64

    start_time = time.time()
    print("\n--- Început Antrenament MLP ---")

    # --- Bucla de Antrenament ---
    for epoca in range(epoci):
        permutare = np.random.permutation(X_antrenare.shape[0])
        X_antrenare_amestecat = X_antrenare[permutare]
        Y_antrenare_amestecat = Y_antrenare[permutare]

        for i in range(0, X_antrenare.shape[0], dimensiune_batch):
            X_batch = X_antrenare_amestecat[i:i + dimensiune_batch]
            Y_batch = Y_antrenare_amestecat[i:i + dimensiune_batch]

            # --- Propagare Înainte ---
            Z1, A1, Z2, A2 = propagare_inainte(X_batch, W1, b1, W2, b2)

            # --- Propagare Înapoi (Backpropagation) ---
            # Eroarea la stratul de ieșire
            dZ2 = A2 - Y_batch

            # Gradienții pentru stratul de ieșire
            grad_W2 = (A1.T @ dZ2) / dimensiune_batch
            grad_b2 = np.sum(dZ2, axis=0) / dimensiune_batch

            # Eroarea propagată la stratul ascuns
            dZ1 = (dZ2 @ W2.T) * derivata_relu(Z1)

            # Gradienții pentru stratul ascuns
            grad_W1 = (X_batch.T @ dZ1) / dimensiune_batch
            grad_b1 = np.sum(dZ1, axis=0) / dimensiune_batch

            # --- Actualizarea Ponderilor ---
            W1 -= rata_invatare * grad_W1
            b1 -= rata_invatare * grad_b1
            W2 -= rata_invatare * grad_W2
            b2 -= rata_invatare * grad_b2

        # Afișarea performanței pe setul de validare
        _, _, _, pred_validare = propagare_inainte(X_validare, W1, b1, W2, b2)
        acuratete = calculeaza_acuratete(pred_validare, Y_validare)
        print(f"Epoca {epoca + 1}/{epoci}, Acuratețe Validare: {acuratete * 100:.2f}%")

    end_time = time.time()
    print(f"\nAntrenament finalizat în {end_time - start_time:.2f} secunde.")

    # --- Generarea Fișierului de Predicții ---
    print("\n--- Generare fișier de predicții pentru Kaggle ---")
    X_test_competitie = incarca_date_competitie('extended_mnist_test.pkl')

    _, _, _, predictii_finale_one_hot = propagare_inainte(X_test_competitie, W1, b1, W2, b2)
    predictii_finale_etichete = np.argmax(predictii_finale_one_hot, axis=1)

    submission_df = pd.DataFrame({
        'ImageId': np.arange(1, len(predictii_finale_etichete) + 1),
        'Label': predictii_finale_etichete
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Fișierul 'submission.csv' a fost generat cu succes!")

