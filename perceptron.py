"""Perceptrón simple con función de activación escalón o sigmoide.

Este módulo no depende de Streamlit: lo usan la app (entrenamiento.py) y los tests.
"""
import numpy as np

# Tablas de verdad de las compuertas lógicas de dos entradas
X_GATE = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
GATES = {
    "OR": np.array([0, 1, 1, 1]),
    "AND": np.array([0, 0, 0, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR (No lineal)": np.array([0, 1, 1, 0]),
}


def gate_data(name):
    """Devuelve (X, y) para la compuerta lógica indicada."""
    return X_GATE, GATES[name]


class Perceptron:
    """
    Perceptrón de dos entradas más bias: z = w0 + w1*x1 + w2*x2.

    - Escalón: salida 1 si z >= 0, si no 0. Se entrena con la regla de Rosenblatt.
    - Sigmoide: salida 1 / (1 + e^-z). Se entrena con descenso de gradiente sobre el
      error cuadrático (regla delta).
    """

    def __init__(self, learning_rate=0.1, n_epochs=10, activation="Escalón", random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation_name = activation
        # Inicialización de pesos aleatoria (reproducible si se fija random_state)
        rng = np.random.default_rng(random_state)
        self.weights = rng.uniform(low=-0.5, high=0.5, size=3)
        self.errors = []

    def _sigmoid(self, z):
        # Clip para evitar overflow en exp
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def activation(self, z):
        if self.activation_name == "Sigmoide":
            return self._sigmoid(z)
        return np.where(z >= 0.0, 1, 0)

    def net_input(self, X):
        """z = w0 + w·x para una entrada (x1, x2) o una matriz de entradas."""
        X = np.atleast_2d(X)
        return self.weights[0] + X @ self.weights[1:]

    def predict_proba(self, X):
        """Valor crudo de la activación (0/1 con escalón, entre 0 y 1 con sigmoide)."""
        return self.activation(self.net_input(X))

    def predict(self, X):
        """Clase predicha. Con sigmoide, salida >= 0.5 equivale a z >= 0."""
        return (self.net_input(X) >= 0.0).astype(int)

    def train(self, X, y):
        """
        Entrena época a época sobre (X, y).

        Es un generador: tras cada época devuelve (época, errores, pesos) para que
        la app pueda animar el entrenamiento.
        """
        X_with_bias = np.insert(X, 0, 1, axis=1)

        for epoch in range(1, self.n_epochs + 1):
            for x, target in zip(X_with_bias, y):
                output = self.activation(x @ self.weights)
                error = target - output

                if self.activation_name == "Sigmoide":
                    # Regla delta: w = w + lr * error * output * (1 - output) * x
                    update = self.learning_rate * error * output * (1 - output) * x
                else:
                    # Regla del Perceptrón (Rosenblatt): w = w + lr * error * x
                    update = self.learning_rate * error * x

                self.weights += update

            # Errores de clasificación al final de la época
            misclassifications = int(np.sum(self.predict(X) != y))
            self.errors.append(misclassifications)

            yield epoch, misclassifications, self.weights.copy()

    def fit(self, X, y):
        """Entrena hasta clasificar todo bien o agotar las épocas."""
        for _, misclassifications, _ in self.train(X, y):
            if misclassifications == 0:
                break
        return self
