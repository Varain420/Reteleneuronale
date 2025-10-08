

import numpy as np
import time
import pickle
import pandas as pd
from torchvision.datasets import MNIST


# --- PASUL 1: Încărcarea și Pregătirea Datelor ---

def descarca_mnist(este_antrenament: bool):

    print(f"Se descarcă setul de date MNIST {'de antrenament' if este_antrenament else 'de validare'}...")
    dataset = MNIST(root='./data',
                    train=este_antrenament,
                    transform=lambda img: np.array(img, dtype=np.float32).flatten(),
                    download=True)

    date_imagini = np.array([img for img, label in dataset])
    date_etichete = np.array([label for img, label in dataset])

    return date_imagini, date_etichete


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
    Funcție specializată și robustă pentru a încărca fișierul de test al competiției.
    Gestionează structura imprevizibilă a fișierelor .pkl.
    """
    print(f"Se încarcă datele de competiție din '{cale_fisier_pkl}'...")
    with open(cale_fisier_pkl, 'rb') as f:
        data_bruta = pickle.load(f, encoding='latin1')

    imagini = data_bruta
    # Navigăm recursiv prin tuple pentru a extrage matricea NumPy
    while isinstance(imagini, tuple):
        imagini = imagini[0]

    # --- LOGICĂ NOUĂ ȘI ROBUSTĂ PENTRU APLATIZARE ---
    numar_total_elemente = imagini.size
    pixeli_per_imagine = 784

    if numar_total_elemente % pixeli_per_imagine != 0:
        raise ValueError("Numărul total de pixeli din fișierul de test nu este un multiplu de 784.")

    nr_imagini = numar_total_elemente // pixeli_per_imagine

    # Forțăm redimensionarea la formatul corect (N, 784)
    imagini_vectorizate = imagini.reshape(nr_imagini, pixeli_per_imagine)

    return imagini_vectorizate / 255.0


# --- PASUL 2: Definirea Funcțiilor Rețelei (Revenire la Perceptron Simplu) ---

def softmax(z):
    """Funcția Softmax stabilă numeric pentru calculul probabilităților."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def propagare_inainte(X, W, b):
    """
    Calculează output-ul rețelei (forward pass) pentru un Perceptron simplu.
    """
    return softmax(X @ W + b)


def calculeaza_acuratete(y_pred_one_hot, y_adevarat_one_hot):
    """Calculează acuratețea comparând predicțiile cu etichetele reale."""
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

    # --- Inițializarea Ponderilor și Hiperparametrilor pentru Perceptron Simplu ---
    nr_caracteristici = X_antrenare.shape[1]
    nr_clase = Y_antrenare.shape[1]

    # Ponderi și bias pentru un singur strat
    W = np.random.randn(nr_caracteristici, nr_clase) * 0.01
    b = np.zeros((1, nr_clase))

    # Hiperparametri ajustați pentru o acuratețe în jur de 92.5%
    rata_invatare_initiala = 0.1
    rata_decay = 0.01  # Scădere lentă a ratei de învățare
    epoci = 50  # Suficiente epoci pentru ca modelul simplu să atingă platoul
    dimensiune_batch = 64

    start_time = time.time()
    print("\n--- Început Antrenament Perceptron Simplu ---")

    # --- Bucla de Antrenament ---
    for epoca in range(epoci):
        # Implementarea "Learning Rate Decay"
        rata_invatare = rata_invatare_initiala / (1 + rata_decay * epoca)

        permutare = np.random.permutation(X_antrenare.shape[0])

        for i in range(0, X_antrenare.shape[0], dimensiune_batch):
            indici = permutare[i:i + dimensiune_batch]
            X_batch, Y_batch = X_antrenare[indici], Y_antrenare[indici]

            # --- Propagare Înainte ---
            A_batch = propagare_inainte(X_batch, W, b)

            # --- Propagare Înapoi (Calculul Gradientului) ---
            dZ = A_batch - Y_batch
            grad_W = (X_batch.T @ dZ) / dimensiune_batch
            grad_b = np.sum(dZ, axis=0) / dimensiune_batch

            # --- Actualizarea Ponderilor (Gradient Descent) ---
            W -= rata_invatare * grad_W
            b -= rata_invatare * grad_b

        # Afișarea performanței pe setul de validare la finalul fiecărei epoci
        pred_validare = propagare_inainte(X_validare, W, b)
        acuratete = calculeaza_acuratete(pred_validare, Y_validare)
        print(f"Epoca {epoca + 1}/{epoci}, Acuratețe Validare: {acuratete * 100:.2f}%")

    end_time = time.time()
    print(f"\nAntrenament finalizat în {end_time - start_time:.2f} secunde.")

    # --- Generarea Fișierului de Predicții pentru Competiție ---
    print("\n--- Generare fișier de predicții pentru Kaggle ---")
    X_test_competitie = incarca_date_competitie('extended_mnist_test.pkl')

    predictii_finale_one_hot = propagare_inainte(X_test_competitie, W, b)
    predictii_finale_etichete = np.argmax(predictii_finale_one_hot, axis=1)

    submission_df = pd.DataFrame({
        'ImageId': np.arange(1, len(predictii_finale_etichete) + 1),
        'Label': predictii_finale_etichete
    })


