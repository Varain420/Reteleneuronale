import re
import math
import sys


# --- Partea 1: Parsarea Sistemului de Ecuații ---

def parseaza_sistem(nume_fisier="equations.txt"):
    A = []
    B = []
    try:
        with open(nume_fisier, 'r') as f:
            for linie in f:
                linie_curata = linie.strip()
                if not linie_curata:  # Ignoră rândurile goale
                    continue

                linie_curata = linie_curata.replace(" ", "")

                # Împarte ecuația la semnul egal
                parti = linie_curata.split('=')
                if len(parti) != 2:
                    continue  # Ignoră liniile malformate
                partea_ecuatiei = parti[0]
                partea_constantei = float(parti[1])
                B.append(partea_constantei)

                # Folosește regex pentru a găsi toți termenii cu coeficienții lor
                coeficienti = []
                for var in ['x', 'y', 'z']:
                    # Găsește termenul corespunzător variabilei curente
                    # Am folosit un raw f-string (fr'') pentru a elimina SyntaxWarning
                    potrivire_termen = re.search(fr'([+-]?\d*\.?\d*){var}', partea_ecuatiei)

                    if potrivire_termen:
                        coef_str = potrivire_termen.group(1)
                        if coef_str == '+':
                            coeficienti.append(1.0)
                        elif coef_str == '-':
                            coeficienti.append(-1.0)
                        elif coef_str == '':
                            # Gestionează cazul 'x' care înseamnă '1x'
                            coeficienti.append(1.0)
                        else:
                            coeficienti.append(float(coef_str))
                    else:
                        # Dacă o variabilă nu este prezentă, coeficientul ei este 0
                        coeficienti.append(0.0)

                A.append(coeficienti)
    except FileNotFoundError:
        print(f"\nEROARE CRITICĂ: Fișierul '{nume_fisier}' nu a fost găsit.")
        print("Asigură-te că fișierul 'equations.txt' se află în același folder cu scriptul Python.")
        sys.exit(1)  # Oprește execuția

    if not A:
        print(f"\nEROARE CRITICĂ: Nu s-au putut citi ecuații din fișierul '{nume_fisier}'.")
        print("Verifică dacă fișierul nu este gol și dacă ecuațiile sunt formatate corect.")
        sys.exit(1)  # Oprește execuția

    return A, B


# --- Partea 2: Operații cu Matrici și Vectori ---

def determinant_3x3(matrice):
    """Calculează determinantul unei matrice 3x3."""
    if len(matrice) != 3 or any(len(rand) != 3 for rand in matrice):
        raise ValueError("Matricea de intrare trebuie să fie 3x3.")

    a11, a12, a13 = matrice[0]
    a21, a22, a23 = matrice[1]
    a31, a32, a33 = matrice[2]

    det = (a11 * (a22 * a33 - a23 * a32) -
           a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31))

    return det


def urma_matrice(matrice):
    """Calculează urma unei matrice pătratice."""
    if len(matrice) != len(matrice[0]):
        raise ValueError("Matricea de intrare trebuie să fie pătratică.")

    return sum(matrice[i][i] for i in range(len(matrice)))


def norma_vector(vector):
    """Calculează norma euclidiană a unui vector."""
    return math.sqrt(sum(x ** 2 for x in vector))


def transpusa(matrice):
    """Calculează transpusa unei matrice."""
    return [[matrice[j][i] for j in range(len(matrice))] for i in range(len(matrice[0]))]


def inmultire_matrice_vector(matrice, vector):
    """Înmulțește o matrice cu un vector."""
    if len(matrice[0]) != len(vector):
        raise ValueError("Numărul de coloane al matricei trebuie să fie egal cu lungimea vectorului.")

    rezultat = []
    for rand in matrice:
        produs_scalar = sum(rand[i] * vector[i] for i in range(len(vector)))
        rezultat.append(produs_scalar)
    return rezultat


# --- Partea 3: Rezolvarea prin Regula lui Cramer ---

def rezolva_cramer(A, B):
    """Rezolvă un sistem de ecuații liniare folosind regula lui Cramer."""
    det_A = determinant_3x3(A)
    if det_A == 0:
        return "Sistemul nu are o soluție unică (determinantul este zero)."

    solutii = []
    for i in range(len(A)):
        # Creează o copie a lui A pentru a o modifica
        A_temp = [rand[:] for rand in A]
        # Înlocuiește coloana i cu vectorul B
        for j in range(len(A_temp)):
            A_temp[j][i] = B[j]

        solutii.append(determinant_3x3(A_temp) / det_A)

    return solutii


# --- Partea 4: Rezolvarea prin Inversare ---

def obtine_minor(matrice, i, j):
    """Returnează minorul matricei prin eliminarea rândului i și coloanei j."""
    return [rand[:j] + rand[j + 1:] for rand in (matrice[:i] + matrice[i + 1:])]


def determinant_2x2(matrice):
    """Calculează determinantul unei matrice 2x2."""
    return matrice[0][0] * matrice[1][1] - matrice[0][1] * matrice[1][0]


def matrice_cofactori(matrice):
    """Calculează matricea de cofactori a unei matrice 3x3."""
    cofactori = []
    for r in range(len(matrice)):
        rand_cofactori = []
        for c in range(len(matrice)):
            minor = obtine_minor(matrice, r, c)
            cofactor = ((-1) ** (r + c)) * determinant_2x2(minor)
            rand_cofactori.append(cofactor)
        cofactori.append(rand_cofactori)
    return cofactori


def matrice_inversa(matrice):
    """Calculează inversa unei matrice 3x3 folosind metoda adjunctei."""
    det = determinant_3x3(matrice)
    if det == 0:
        raise ValueError("Matricea este singulară și nu poate fi inversată.")

    cofactori = matrice_cofactori(matrice)
    adjuncta = transpusa(cofactori)

    inversa = [[elem / det for elem in rand] for rand in adjuncta]

    return inversa


def rezolva_prin_inversare(A, B):
    """Rezolvă un sistem de ecuații liniare prin inversarea matricei."""
    try:
        A_inv = matrice_inversa(A)
        return inmultire_matrice_vector(A_inv, B)
    except ValueError as e:
        return str(e)


# --- Execuția Principală ---
if __name__ == "__main__":

    print("--- Partea 1: Parsarea Sistemului ---")
    A, B = parseaza_sistem()
    print("Matricea A:")
    for rand in A:
        print(f"  {rand}")
    print("\nVectorul B:")
    print(f"  {B}\n")

    print("--- Partea 2: Operații cu Matrici și Vectori ---")
    det_A = determinant_3x3(A)
    print(f"Determinantul lui A: {det_A}")

    urma_A = urma_matrice(A)
    print(f"Urma lui A: {urma_A}")

    norma_B = norma_vector(B)
    print(f"Norma Euclidiană a lui B: {norma_B:.4f}")

    A_T = transpusa(A)
    print("\nTranspusa lui A:")
    for rand in A_T:
        print(f"  {rand}")

    try:
        AxB = inmultire_matrice_vector(A, B)
        print("\nÎnmulțire Matrice-Vector (A * B):")
        print(f"  {AxB}\n")
    except ValueError as e:
        print(f"\nEroare la înmulțirea Matrice-Vector: {e}\n")

    print("--- Partea 3: Rezolvarea cu Regula lui Cramer ---")
    solutie_cramer = rezolva_cramer(A, B)
    if isinstance(solutie_cramer, list):
        print(f"Soluție (x, y, z): ({solutie_cramer[0]:.4f}, {solutie_cramer[1]:.4f}, {solutie_cramer[2]:.4f})\n")
    else:
        print(f"{solutie_cramer}\n")

    print("--- Partea 4: Rezolvarea prin Inversarea Matricei ---")
    solutie_inversare = rezolva_prin_inversare(A, B)
    if isinstance(solutie_inversare, list):
        print(
            f"Soluție (x, y, z): ({solutie_inversare[0]:.4f}, {solutie_inversare[1]:.4f}, {solutie_inversare[2]:.4f})")
        print("\nInversa lui A (A^-1):")
        A_inv = matrice_inversa(A)
        for rand in A_inv:
            print(f"  [{', '.join(f'{x:.4f}' for x in rand)}]")
    else:
        print(solutie_inversare)


