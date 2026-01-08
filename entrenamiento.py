import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Configuración de la página
st.set_page_config(page_title="Entrenamiento del Perceptrón", page_icon= '🧠', layout="wide")

class Perceptron:
    """
    Implementación del Perceptrón capaz de usar función Umbral o Sigmoide.
    """
    def __init__(self, learning_rate=0.1, n_epochs=10, activation="Escalón"):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation_name = activation
        # Inicialización de pesos aleatoria
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=3)
        self.errors = []

    def _sigmoid(self, z):
        # Clip para evitar overflow en exp
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _activation_function(self, net_input):
        if self.activation_name == "Sigmoide":
            return self._sigmoid(net_input)
        else: # Umbral (Step)
            return np.where(net_input >= 0.0, 1, 0)

    def predict(self, inputs):
        """Predice la clase para una entrada dada."""
        # Aseguramos que inputs sea numpy array
        inputs = np.array(inputs)
        # Añade el bias x0 = 1
        inputs_with_bias = np.insert(inputs, 0, 1)
        net_input = np.dot(inputs_with_bias, self.weights)
        
        output = self._activation_function(net_input)
        
        # Para sigmoide, convertimos probabilidad a clase para la métrica de error
        if self.activation_name == "Sigmoide":
            return 1 if output >= 0.5 else 0
        return output

    def predict_proba(self, inputs):
        """Para uso interno: devuelve el valor crudo de activación."""
        inputs = np.array(inputs)
        inputs_with_bias = np.insert(inputs, 0, 1)
        net_input = np.dot(inputs_with_bias, self.weights)
        return self._activation_function(net_input)

    def train_step(self, X, y):
        """
        Ejecuta UNA época de entrenamiento y devuelve los errores.
        Es un generador para permitir la animación en Streamlit.
        """
        X_train_with_bias = np.insert(X, 0, 1, axis=1)
        
        for epoch in range(self.n_epochs):
            misclassifications = 0
            
            for inputs_with_bias, target in zip(X_train_with_bias, y):
                net_input = np.dot(inputs_with_bias, self.weights)
                output = self._activation_function(net_input)
                
                # Cálculo del error
                error = target - output
                
                # Regla de actualización de pesos
                if self.activation_name == "Sigmoide":
                    # Gradiente descendente para Sigmoide (aprox)
                    # w = w + lr * error * output * (1-output) * x
                    update = self.learning_rate * error * output * (1 - output) * inputs_with_bias
                else:
                    # Regla estándar del Perceptrón (Rosenblatt)
                    # w = w + lr * error * x
                    update = self.learning_rate * error * inputs_with_bias
                
                self.weights += update

            # Calcular errores de clasificación (discretos) al final de la época
            for xi, target in zip(X, y):
                if self.predict(xi) != target:
                    misclassifications += 1
            
            self.errors.append(misclassifications)
            
            # Devolvemos el estado actual para graficar
            yield epoch + 1, misclassifications, self.weights

# --- Interfaz de Streamlit ---

st.title("🧠 Visualizador de Entrenamiento del Perceptrón")
st.markdown("""
Esta aplicación permite visualizar cómo un **Perceptrón Simple** aprende a separar clases linealmente.
Puedes experimentar con diferentes compuertas lógicas y funciones de activación.
""")

# Sidebar de configuración
st.sidebar.header("⚙️ Configuración")

gate_type = st.sidebar.selectbox(
    "Compuerta Lógica (Datos)",
    ("OR", "AND", "NAND", "XOR (No lineal)"),
    index=0
)

activation_type = st.sidebar.selectbox(
    "Función de Activación",
    ("Escalón", "Sigmoide"),
    index=1,
    help="Escalón: Salida binaria 0 o 1. Sigmoide: Salida continua entre 0 y 1."
)

lr = st.sidebar.slider("Tasa de Aprendizaje (Learning Rate)", 0.01, 1.0, 0.5, 0.01)
epochs = st.sidebar.slider("Número de Épocas", 1, 1000, 15)
anim_speed = st.sidebar.slider("Velocidad de Animación (seg)", 0.1, 2.0, 0.5)

# Definición de Datos
if gate_type == "OR":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    st.sidebar.success("Target: [0, 1, 1, 1]")
elif gate_type == "AND":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    st.sidebar.success("Target: [0, 0, 0, 1]")
elif gate_type == "NAND":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 1, 1, 0])
    st.sidebar.success("Target: [1, 1, 1, 0]")
else: # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    st.sidebar.warning("Target: [0, 1, 1, 0] (XOR no es linealmente separable)")

# Botón de inicio
start_btn = st.sidebar.button("▶️ Iniciar Entrenamiento", type="primary")

# Contenedores para gráficos y métricas
col1, col2 = st.columns([3, 1])
plot_placeholder = col1.empty()
metrics_placeholder = col2.empty()

# Función para graficar
def plot_boundary(X, y, weights, epoch, errors, activation):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Datos
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', s=150, label='Clase 0', edgecolors='black')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='X', s=150, label='Clase 1', edgecolors='black')
    
    # Frontera
    w0, w1, w2 = weights
    x_min, x_max = -1.5, 3.5
    y_min, y_max = -1.5, 3.5
    
    if w2 != 0:
        x_vals = np.array([x_min, x_max])
        # w0 + w1*x + w2*y = 0  =>  y = -(w0 + w1*x) / w2
        y_vals = -(w0 + w1 * x_vals) / w2
        ax.plot(x_vals, y_vals, 'g--', linewidth=3, label='Frontera de Decisión')
    else:
        # Si w2 es 0, la línea es vertical x = -w0/w1
        if w1 != 0:
            x_val = -w0 / w1
            ax.axvline(x=x_val, color='g', linestyle='--', linewidth=3, label='Frontera de Decisión')

    # Región sombreada (solo para visualización estética)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Calculamos predicción para todo el meshgrid para colorear el fondo
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Truco para vectorizar la predicción sin instanciar la clase completa de nuevo
    grid_bias = np.insert(grid_points, 0, 1, axis=1)
    Z = np.dot(grid_bias, weights)
    Z = np.where(Z >= 0, 1, 0)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.1, cmap='coolwarm')

    ax.set_title(f'Frontera de Decisión - Época {epoch}', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return fig

# Lógica de Ejecución
if start_btn:
    # Reiniciar modelo
    model = Perceptron(learning_rate=lr, n_epochs=epochs, activation=activation_type)
    
    # Barra de progreso
    progress_bar = st.progress(0)
    
    # Loop de entrenamiento animado
    for epoch, errors, weights in model.train_step(X, y):
        
        # Actualizar gráfico
        fig = plot_boundary(X, y, weights, epoch, errors, activation_type)
        plot_placeholder.pyplot(fig)
        plt.close(fig) # Liberar memoria
        
        # Actualizar métricas
        with metrics_placeholder.container():
            st.metric(label="Época Actual", value=f"{epoch}/{epochs}")
            st.metric(label="Errores de Clasificación", value=errors, delta=-errors if epoch > 1 else None)
            
            st.markdown("### Pesos Sinápticos")
            st.code(f"Bias (w0): {weights[0]:.4f}\n"
                    f"w1:        {weights[1]:.4f}\n"
                    f"w2:        {weights[2]:.4f}")
            
            if errors == 0:
                st.success("✅ ¡Convergencia Alcanzada!")
                progress_bar.progress(1.0)
                break
        
        # Actualizar barra
        progress_bar.progress(epoch / epochs)
        
        # Pausa para animación
        time.sleep(anim_speed)
    
    if model.errors[-1] > 0:
        st.error(f"❌ El entrenamiento finalizó sin convergencia perfecta ({model.errors[-1]} errores).")
        if gate_type == "XOR (No lineal)":
            st.info("💡 Nota: El problema XOR no es linealmente separable, por lo que un Perceptrón simple nunca podrá resolverlo perfectamente.")

else:
    # Estado inicial (antes de pulsar botón)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', s=150, label='Clase 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='X', s=150, label='Clase 1')
    ax.set_title("Datos Iniciales", fontsize=16)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plot_placeholder.pyplot(fig)
    
    metrics_placeholder.info("Configura los parámetros en el menú de la izquierda y pulsa 'Iniciar Entrenamiento'.")