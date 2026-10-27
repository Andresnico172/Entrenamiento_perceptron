import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from perceptron import Perceptron, gate_data

SEPARABLE = ["OR", "AND", "NAND"]


@pytest.mark.parametrize("gate", SEPARABLE)
def test_escalon_converge_en_compuertas_separables(gate):
    X, y = gate_data(gate)
    model = Perceptron(learning_rate=0.5, n_epochs=100, activation="Escalón", random_state=0).fit(X, y)
    assert model.errors[-1] == 0
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("gate", SEPARABLE)
def test_sigmoide_converge_en_compuertas_separables(gate):
    X, y = gate_data(gate)
    model = Perceptron(learning_rate=1.0, n_epochs=2000, activation="Sigmoide", random_state=0).fit(X, y)
    assert model.errors[-1] == 0
    proba = model.predict_proba(X)
    assert np.all((proba > 0) & (proba < 1))


@pytest.mark.parametrize("activation", ["Escalón", "Sigmoide"])
def test_xor_nunca_converge(activation):
    X, y = gate_data("XOR (No lineal)")
    model = Perceptron(learning_rate=0.5, n_epochs=500, activation=activation, random_state=0).fit(X, y)
    assert len(model.errors) == 500
    assert min(model.errors) >= 1


def test_semilla_reproducible():
    X, y = gate_data("AND")
    a = Perceptron(n_epochs=5, random_state=7)
    b = Perceptron(n_epochs=5, random_state=7)
    for (_, ea, wa), (_, eb, wb) in zip(a.train(X, y), b.train(X, y)):
        assert ea == eb
        np.testing.assert_array_equal(wa, wb)


def test_generador_devuelve_una_tupla_por_epoca():
    X, y = gate_data("XOR (No lineal)")
    steps = list(Perceptron(n_epochs=3, random_state=0).train(X, y))
    assert [epoch for epoch, _, _ in steps] == [1, 2, 3]


def test_app_arranca_y_entrena():
    at = AppTest.from_file("entrenamiento.py", default_timeout=30).run()
    assert not at.exception
    at.sidebar.slider[2].set_value(0.0)  # sin pausa de animación
    at.sidebar.button[0].click().run()
    assert not at.exception
    # Con los valores por defecto (OR, sigmoide) el entrenamiento debe converger
    assert any("Convergencia alcanzada" in s.value for s in at.main.success)
    assert not at.main.error
