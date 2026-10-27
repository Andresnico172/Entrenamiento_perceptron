import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from perceptron import GATES, Perceptron, gate_data

# Configuración de la página
st.set_page_config(page_title="Entrenamiento del Perceptrón", page_icon='🧠', layout="wide")

# Límites de los gráficos (iguales antes y durante el entrenamiento)
X_MIN, X_MAX = -0.5, 1.5
Y_MIN, Y_MAX = -0.5, 1.5

# --- Interfaz de Streamlit ---

st.title("🧠 Visualizador de Entrenamiento del Perceptrón")
st.markdown("""
Esta aplicación permite visualizar cómo un **Perceptrón Simple** aprende a separar clases linealmente.
Puedes experimentar con diferentes compuertas lógicas y funciones de activación.
""")

with st.expander("📖 ¿Cómo funciona?"):
    st.markdown(r"""
El perceptrón calcula una combinación lineal de las entradas más un sesgo (bias) y la pasa por una función de activación:

$$z = w_0 + w_1 x_1 + w_2 x_2, \qquad \hat y = f(z)$$

- **Escalón:** $f(z) = 1$ si $z \ge 0$ y $0$ en otro caso. Pesos actualizados con la **regla de Rosenblatt**:
  $w \leftarrow w + \eta\,(y - \hat y)\,x$.
- **Sigmoide:** $f(z) = 1/(1+e^{-z})$. Pesos actualizados por **descenso de gradiente** del error cuadrático (regla delta):
  $w \leftarrow w + \eta\,(y - \hat y)\,\hat y\,(1-\hat y)\,x$. La clase predicha es 1 si $\hat y \ge 0.5$.

La **frontera de decisión** es la recta $w_0 + w_1 x_1 + w_2 x_2 = 0$. Si los datos son linealmente separables
(AND, OR, NAND), el teorema de convergencia del perceptrón garantiza que la regla de Rosenblatt encuentra una
frontera en un número finito de pasos. **XOR** no es linealmente separable: ninguna recta lo resuelve y hace falta
una red con capa oculta.
""")

# Sidebar de configuración
st.sidebar.header("⚙️ Configuración")

gate_type = st.sidebar.selectbox("Compuerta Lógica (Datos)", list(GATES), index=0)

activation_type = st.sidebar.selectbox(
    "Función de Activación",
    ("Escalón", "Sigmoide"),
    index=1,
    help="Escalón: Salida binaria 0 o 1. Sigmoide: Salida continua entre 0 y 1."
)

lr = st.sidebar.slider("Tasa de Aprendizaje (Learning Rate)", 0.01, 1.0, 0.5, 0.01)
epochs = st.sidebar.slider("Número de Épocas", 1, 1000, 50,
                           help="El entrenamiento se detiene antes si se alcanza 0 errores.")
anim_speed = st.sidebar.slider("Velocidad de Animación (seg)", 0.0, 2.0, 0.2)
seed = st.sidebar.number_input("Semilla aleatoria (pesos iniciales)", min_value=0, max_value=9999, value=42,
                               help="Con la misma semilla, el entrenamiento es reproducible.")

# Definición de Datos
X, y = gate_data(gate_type)
if gate_type.startswith("XOR"):
    st.sidebar.warning(f"Target: {y.tolist()} (XOR no es linealmente separable)")
else:
    st.sidebar.success(f"Target: {y.tolist()}")

# Botón de inicio
start_btn = st.sidebar.button("▶️ Iniciar Entrenamiento", type="primary")

# Contenedores para gráficos y métricas
col1, col2 = st.columns([3, 1])
plot_placeholder = col1.empty()
metrics_placeholder = col2.empty()
curve_placeholder = st.empty()


def plot_data(ax, X, y):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', s=150, label='Clase 0', edgecolors='black')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='X', s=150, label='Clase 1', edgecolors='black')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, linestyle=':', alpha=0.6)


# Función para graficar
def plot_boundary(X, y, model, epoch):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Región sombreada: salida de la activación en cada punto del plano
    xx, yy = np.meshgrid(np.linspace(X_MIN, X_MAX, 200), np.linspace(Y_MIN, Y_MAX, 200))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    levels = np.linspace(0, 1, 21) if model.activation_name == "Sigmoide" else [-0.5, 0.5, 1.5]
    ax.contourf(xx, yy, Z, levels=levels, alpha=0.25, cmap='coolwarm_r', vmin=0, vmax=1)

    # Frontera: w0 + w1*x + w2*y = 0
    w0, w1, w2 = model.weights
    if w2 != 0:
        x_vals = np.array([X_MIN, X_MAX])
        y_vals = -(w0 + w1 * x_vals) / w2
        ax.plot(x_vals, y_vals, 'g--', linewidth=3, label='Frontera de Decisión')
    elif w1 != 0:
        # Si w2 es 0, la línea es vertical x = -w0/w1
        ax.axvline(x=-w0 / w1, color='g', linestyle='--', linewidth=3, label='Frontera de Decisión')

    plot_data(ax, X, y)
    ax.set_title(f'Frontera de Decisión - Época {epoch}', fontsize=16)
    ax.legend(loc='upper right')

    return fig


# Lógica de Ejecución
if start_btn:
    # Reiniciar modelo
    model = Perceptron(learning_rate=lr, n_epochs=epochs, activation=activation_type, random_state=int(seed))

    # Barra de progreso
    progress_bar = st.progress(0)

    # Loop de entrenamiento animado
    for epoch, errors, weights in model.train(X, y):

        # Actualizar gráfico
        fig = plot_boundary(X, y, model, epoch)
        plot_placeholder.pyplot(fig)
        plt.close(fig)  # Liberar memoria

        # Actualizar métricas
        with metrics_placeholder.container():
            st.metric(label="Época Actual", value=f"{epoch}/{epochs}")
            prev_errors = model.errors[-2] if epoch > 1 else None
            st.metric(label="Errores de Clasificación", value=errors,
                      delta=errors - prev_errors if prev_errors is not None else None, delta_color="inverse")

            st.markdown("### Pesos Sinápticos")
            st.code(f"Bias (w0): {weights[0]:.4f}\n"
                    f"w1:        {weights[1]:.4f}\n"
                    f"w2:        {weights[2]:.4f}")

        # Curva de errores por época
        with curve_placeholder.container():
            st.markdown("#### Errores de clasificación por época")
            st.line_chart(pd.DataFrame({"Errores": model.errors}, index=range(1, epoch + 1)), height=200)

        # Actualizar barra
        progress_bar.progress(epoch / epochs)

        if errors == 0:
            progress_bar.progress(1.0)
            break

        # Pausa para animación
        time.sleep(anim_speed)

    if model.errors[-1] == 0:
        st.success(f"✅ ¡Convergencia alcanzada en la época {len(model.errors)}!")
    else:
        st.error(f"❌ El entrenamiento finalizó sin convergencia perfecta ({model.errors[-1]} errores).")
        if gate_type.startswith("XOR"):
            st.info("💡 Nota: El problema XOR no es linealmente separable, por lo que un Perceptrón simple nunca "
                    "podrá resolverlo perfectamente.")

    # Tabla de verdad con las predicciones finales
    st.markdown("#### Resultado sobre la tabla de verdad")
    st.dataframe(pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "Objetivo": y,
        "Activación": np.round(model.predict_proba(X), 4),
        "Predicción": model.predict(X),
        "¿Correcto?": np.where(model.predict(X) == y, "✅", "❌"),
    }), hide_index=True)

else:
    # Estado inicial (antes de pulsar botón)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data(ax, X, y)
    ax.set_title("Datos Iniciales", fontsize=16)
    ax.legend()
    plot_placeholder.pyplot(fig)
    plt.close(fig)

    metrics_placeholder.info("Configura los parámetros en el menú de la izquierda y pulsa 'Iniciar Entrenamiento'.")
