import streamlit as st
from PIL import Image
import utils

# Configuración de página
st.set_page_config(page_title="Pokémon Classifier AI", page_icon="🔮")
st.title("🔮 Clasificador de Tipos de Pokémon")
st.write("Subí la foto de un Pokémon de la Generación 1 y la IA te dirá su tipo principal.")

# Ruta del modelo en el mismo directorio
RUTA_MEJOR_MODELO = "best_model.pt"

# Carga cacheada del modelo utilizando Streamlit
@st.cache_resource
def get_model():
    return utils.cargar_modelo_ganador(RUTA_MEJOR_MODELO)

modelo, mensaje, carga_exitosa = get_model()

# Mostrar estado de la carga en la barra lateral
if carga_exitosa:
    st.sidebar.success(mensaje)
else:
    st.sidebar.error(mensaje)
    st.stop() # Frena la app si no hay modelo

# Subida de archivos
uploaded_file = st.file_uploader("Elegí una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    imagen = Image.open(uploaded_file).convert("RGB")
    st.image(imagen, caption='Imagen ingresada', use_container_width=True)
    
    with st.spinner("🔮 Clasificando..."):
        # Llamada a las funciones de utilidades
        pred_idx, probabilidades, clases = utils.predecir_imagen(modelo, imagen)
        
    clase_predicha = clases[pred_idx]
    confianza = probabilidades[pred_idx].item() * 100
    
    # Resultados principales
    st.subheader(f"Resultado: Tipo **{clase_predicha}**")
    st.progress(confianza / 100)
    st.write(f"Probabilidad de acierto del tipo principal: **{confianza:.2f}%**")
    
    # --- Agregado para cumplir con "probabilidades por clase" ---
    st.write("---")
    st.subheader("📊 Distribución de probabilidades:")
    
    # Crear un diccionario ordenado para mostrar en un gráfico de barras de Streamlit
    diccionario_probabilidades = {clases[i]: float(probabilidades[i]) for i in range(len(clases))}
    st.bar_chart(diccionario_probabilidades)