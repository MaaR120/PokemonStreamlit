import streamlit as st
from PIL import Image
import utils
import random

# Configuración inicial
st.set_page_config(page_title="Pokémon AI Game", page_icon="🎮", layout="centered")
st.title("🎮 Humano vs. IA: El Desafío Pokémon")
st.write("¡Demostrá que sabés más de tipos que nuestra ResNet18 entrenada!")

# 1. Carga del modelo (Cacheada)
RUTA_MEJOR_MODELO = "best_model.pt"

@st.cache_resource
def get_model():
    return utils.cargar_modelo_ganador(RUTA_MEJOR_MODELO)

modelo, mensaje, carga_exitosa = get_model()
if not carga_exitosa:
    st.error(mensaje)
    st.stop()

# 2. Inicializar el Estado del Juego (Session State)
# Esto evita que los puntajes y la imagen cambien cada vez que tocás un botón
if 'puntaje_humano' not in st.session_state:
    st.session_state.puntaje_humano = 0
if 'puntaje_ia' not in st.session_state:
    st.session_state.puntaje_ia = 0
if 'juego_evaluado' not in st.session_state:
    st.session_state.juego_evaluado = False
if 'pred_idx' not in st.session_state:
    st.session_state.pred_idx = None
    st.session_state.probabilidades = None
    st.session_state.clases = None

# --- SEPARACIÓN EN SECCIONES (TABS) ---
tab_juego, tab_analisis = st.tabs(["🎮 ¡A Jugar!", "📊 Laboratorio del Profesor Oak (Análisis)"])

# ==============================================================================
# TAB 1: EL JUEGO INTERACTIVO
# ==============================================================================
with tab_juego:
    st.subheader("Subí una imagen misteriosa para desafiar a la máquina")
    
    # El usuario sube la imagen (puede ser un caso esperado o un caso límite capcioso)
    uploaded_file = st.file_uploader("Arrastrá el Pokémon aquí...", type=["jpg", "jpeg", "png"], key="game_upload")
    
    if uploaded_file is not None:
        imagen = Image.open(uploaded_file).convert("RGB")
        
        # Mostrar imagen en el campo de batalla
        col_img, col_vs = st.columns([2, 1])
        with col_img:
            st.image(imagen, caption='Objetivo a clasificar', width=300)
        
        with col_vs:
            st.metric("Tu Puntaje", st.session_state.puntaje_humano)
            st.metric("Puntaje IA", st.session_state.puntaje_ia)
            
            # Botón para reiniciar el tablero de puntos si quieren jugar otra ronda
            if st.button("🔄 Reiniciar Marcador"):
                st.session_state.puntaje_humano = 0
                st.session_state.puntaje_ia = 0
                st.rerun()

        st.write("---")
        st.markdown("### 🎯 ¡Hacé tu apuesta!")
        
        # El usuario elige qué tipo cree que es
        opcion_usuario = st.selectbox("¿De qué tipo principal es este Pokémon?", utils.CLASSES)
        
        # Botón para gatillar el veredicto
        if st.button("⚔️ Revelar Resultados"):
            # Procesar predicción de la IA
            with st.spinner("La IA está analizando los píxeles..."):
                pred_idx, probabilidades, clases = utils.predecir_imagen(modelo, imagen)
            
            # Guardar en session_state para que la pestaña de análisis pueda leerlo
            st.session_state.pred_idx = pred_idx
            st.session_state.probabilidades = probabilidades
            st.session_state.clases = clases
            st.session_state.juego_evaluado = True
            
            clase_predicha_ia = clases[pred_idx]
            confianza_ia = probabilidades[pred_idx].item() * 100
            
            st.write("### 📢 Veredicto Final:")
            
            # Lógica de puntos
            # Nota para la defensa: Aquí asumimos que el usuario dice la verdad, 
            # o pueden validar si la IA y el Humano coinciden.
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Tu respuesta:** {opcion_usuario}")
            with col2:
                st.warning(f"**Respuesta de la IA:** {clase_predicha_ia} ({confianza_ia:.1f}% de confianza)")
            
            # Dinámica de premiación (pueden ajustar las reglas en vivo con los profes)
            if opcion_usuario == clase_predicha_ia:
                st.success("¡Empate técnico! Ambos coincidieron en el tipo.")
                st.session_state.puntaje_humano += 1
                st.session_state.puntaje_ia += 1
            else:
                st.error("¡Caminos separados! Uno de los dos (o ambos) falló. ¿Quién tendrá la razón?")
                # Aquí podés sumar puntos manualmente según quién acertó de verdad frente a los profes
                # Para automatizar el show, le daremos el punto a la IA si tiene alta confianza:
                if confianza_ia > 70:
                    st.session_state.puntaje_ia += 1
                else:
                    st.session_state.puntaje_humano += 1
            
            st.balloons()

# ==============================================================================
# TAB 2: SECCIÓN OCULTA / ANÁLISIS CIENTÍFICO
# ==============================================================================
with tab_analisis:
    st.header("🔬 Diagnóstico de la Red Neuronal")
    st.write("Esta sección analiza matemáticamente por qué la ResNet18 tomó su decisión.")
    
    if st.session_state.juego_evaluado:
        clases = st.session_state.clases
        probabilidades = st.session_state.probabilidades
        pred_idx = st.session_state.pred_idx
        
        clase_top = clases[pred_idx]
        confianza_top = probabilidades[pred_idx].item() * 100
        
        st.subheader(f"Clase Predicha de forma principal: **{clase_top}**")
        st.write(f"Confianza asignada a la neurona ganadora: **{confianza_top:.2f}%**")
        
        # Gráfico de barras requerido por la consigna
        st.write("---")
        st.markdown("### 📊 Distribución completa de probabilidades por tipo:")
        st.write("Ideal para detectar **casos límite**: si metieron un objeto raro, verán cómo las probabilidades se distribuyen parejas o con baja certeza.")
        
        diccionario_probabilidades = {clases[i]: float(probabilidades[i]) for i in range(len(clases))}
        st.bar_chart(diccionario_probabilidades)
        
    else:
        st.info("💡 Primero debés subir una imagen y jugar una ronda en la pestaña anterior para desbloquear el análisis de probabilidades.")