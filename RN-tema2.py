import numpy as np


# --- Clasa Perceptron Multi-Clasă ---

class PerceptronMultiClasa:
    def __init__(self, nr_intrari, nr_iesiri, rata_invatare=0.1):
        """
        Inițializează rețeaua.

        Args:
            nr_intrari (int): Numărul de caracteristici de intrare (784 pentru MNIST).
            nr_iesiri (int): Numărul de clase de ieșire (10 pentru cifre).
            rata_invatare (float): Rata de învățare pentru actualizarea ponderilor.
        """
        self.rata_invatare = rata_invatare
        # Inițializăm ponderile W cu valori mici, aleatorii. Transpunem pentru a se potrivi cu formula.
        # Forma va fi (nr_iesiri, nr_intrari) pentru a facilita înmulțirea cu X transpus.
        self.ponderi = np.random.randn(nr_iesiri, nr_intrari) * 0.01
        # Inițializăm bias-urile b cu zero.
        self.bias = np.zeros((nr_iesiri, 1))

    def softmax(self, z):
        """
        Calculează funcția de activare softmax.
        Scădem max(z) pentru stabilitate numerică (previne valori foarte mari la exponențiere).
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def propagare_inainte(self, X):
        """
        Realizează pasul de propagare înainte.

        Args:
            X (np.array): Matricea de intrare de formă (nr_caracteristici, nr_exemple).

        Returns:
            np.array: Probabilitățile de ieșire (predicțiile) de formă (nr_clase, nr_exemple).
        """
        # z = W * x + b
        z = np.dot(self.ponderi, X) + self.bias
        # y_pred = softmax(z)
        A = self.softmax(z)
        return A

    def calculeaza_pierdere(self, A, Y):
        """
        Calculează pierderea cross-entropy.

        Args:
            A (np.array): Predicțiile modelului (ieșirea softmax).
            Y (np.array): Etichetele reale, în format one-hot encoding.

        Returns:
            float: Valoarea pierderii.
        """
        m = Y.shape[1]  # Numărul de exemple
        # Adăugăm o valoare mică (1e-8) pentru a evita log(0)
        pierdere = -np.sum(Y * np.log(A + 1e-8)) / m
        return pierdere

    def propagare_inapoi_si_actualizare(self, X, Y, A):
        """
        Calculează gradienții și actualizează ponderile și bias-urile.

        Args:
            X (np.array): Matricea de intrare.
            Y (np.array): Etichetele reale (one-hot).
            A (np.array): Predicțiile modelului.
        """
        m = X.shape[1]  # Numărul de exemple

        # Calculul gradientului pentru cross-entropy cu softmax este surprinzător de simplu:
        dZ = A - Y

        # Calculul gradienților pentru ponderi și bias
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Actualizarea ponderilor și bias-urilor
        self.ponderi = self.ponderi - self.rata_invatare * dW
        self.bias = self.bias - self.rata_invatare * db

    def antrenare(self, X_antrenare, Y_antrenare, nr_epoci):
        """
        Orchestrează procesul de antrenare a modelului.
        """
        print("Început antrenament...")
        for epoca in range(nr_epoci):
            # 1. Propagare înainte
            A = self.propagare_inainte(X_antrenare)

            # 2. Calculează pierderea (opțional, pentru monitorizare)
            pierdere = self.calculeaza_pierdere(A, Y_antrenare)

            # 3. Propagare înapoi și actualizare
            self.propagare_inapoi_si_actualizare(X_antrenare, Y_antrenare, A)

            if (epoca + 1) % 100 == 0:
                print(f'> Epoca={epoca + 1}, Pierdere={pierdere:.4f}')
        print("Antrenament finalizat.")

    def predictie(self, X):
        """
        Realizează o predicție pe date noi.
        Returnează clasa cu cea mai mare probabilitate.
        """
        A = self.propagare_inainte(X)
        predictii = np.argmax(A, axis=0)
        return predictii


# --- Execuția Principală ---
if __name__ == '__main__':
    # Crearea unui set de date SIMULAT pentru a testa implementarea
    # Acestea ar trebui înlocuite cu datele reale (ex: MNIST)
    NR_EXEMPLE = 1000
    NR_CARACTERISTICI = 784  # 28x28 pixeli
    NR_CLASE = 10  # Cifrele 0-9

    # Generăm date de intrare aleatorii
    X_antrenare = np.random.rand(NR_CARACTERISTICI, NR_EXEMPLE)

    # Generăm etichete reale aleatorii și le convertim în one-hot encoding
    etichete_reale = np.random.randint(0, NR_CLASE, NR_EXEMPLE)
    Y_antrenare = np.zeros((NR_CLASE, NR_EXEMPLE))
    Y_antrenare[etichete_reale, np.arange(NR_EXEMPLE)] = 1

    print(f"Forma datelor de intrare X: {X_antrenare.shape}")
    print(f"Forma etichetelor Y (one-hot): {Y_antrenare.shape}")

    # Definirea parametrilor
    rata_invatare = 0.05
    nr_epoci = 1000

    # Crearea și antrenarea modelului
    perceptron = PerceptronMultiClasa(nr_intrari=NR_CARACTERISTICI, nr_iesiri=NR_CLASE, rata_invatare=rata_invatare)
    perceptron.antrenare(X_antrenare, Y_antrenare, nr_epoci)

    # Testarea pe un exemplu
    X_test = np.random.rand(NR_CARACTERISTICI, 5)  # 5 exemple noi
    predictii = perceptron.predictie(X_test)
    print(f"\nPredicții pe 5 exemple noi: {predictii}")
