import streamlit as st
import utils

st.set_page_config(page_title="Pokémon AI Game", page_icon="🎮", layout="centered")
st.title("🎮 Humano vs. IA: El Desafío Pokémon")
st.write("¡Competí contra la ResNet18! ¿Quién descubrirá el tipo verdadero primero?")

# 1. Carga del modelo
RUTA_MEJOR_MODELO = "best_model.pt"

@st.cache_resource
def get_model():
    return utils.cargar_modelo_ganador(RUTA_MEJOR_MODELO)

modelo, mensaje, carga_exitosa = get_model()
if not carga_exitosa:
    st.error(mensaje)
    st.stop()

# 2. Inicializar el Estado del Juego
if 'puntaje_humano' not in st.session_state:
    st.session_state.puntaje_humano = 0
if 'puntaje_ia' not in st.session_state:
    st.session_state.puntaje_ia = 0

# Variables del turno actual
if 'pkmn_nombre' not in st.session_state:
    # Cargar el primer Pokémon al iniciar la app
    nombre, tipo, img = utils.obtener_pokemon_aleatorio()
    st.session_state.pkmn_nombre = nombre
    st.session_state.pkmn_tipo_real = tipo
    st.session_state.pkmn_imagen = img
    st.session_state.turno_evaluado = False
    st.session_state.pred_idx = None
    st.session_state.probabilidades = None

# Función para pasar al siguiente turno
def siguiente_turno():
    nombre, tipo, img = utils.obtener_pokemon_aleatorio()
    st.session_state.pkmn_nombre = nombre
    st.session_state.pkmn_tipo_real = tipo
    st.session_state.pkmn_imagen = img
    st.session_state.turno_evaluado = False
    st.session_state.pred_idx = None
    st.session_state.probabilidades = None

# --- ESTRUCTURA DE TABS ---
tab_juego, tab_analisis = st.tabs(["🎮 ¡A Jugar!", "📊 Laboratorio de Probabilidades"])

# ==============================================================================
# TAB 1: EL CAMPO DE BATALLA
# ==============================================================================
with tab_juego:
    # Fila superior: Marcadores y Control
    col_score1, col_score2, col_btn = st.columns([1, 1, 1.5])
    with col_score1:
        st.metric("🏆 Tu Puntaje", st.session_state.puntaje_humano)
    with col_score2:
        st.metric("🤖 Puntaje IA", st.session_state.puntaje_ia)
    with col_btn:
        st.write("") # Espacio estético
        if st.button("⏭️ Siguiente Pokémon", use_container_width=True):
            siguiente_turno()
            st.rerun()

    st.write("---")
    
    # Mostrar la imagen del rival misterioso
    st.subheader(f"¿Quién es este Pokémon?")
    st.image(st.session_state.pkmn_imagen, width=280)
    
    # Selección de respuesta del jugador
    # Deshabilitamos el selector si ya se evaluó para evitar trampas en el mismo turno
    opcion_usuario = st.selectbox(
        "Elegí el tipo principal:", 
        utils.CLASSES, 
        disabled=st.session_state.turno_evaluado
    )
    
    # Botón de acción principal
    if not st.session_state.turno_evaluado:
        if st.button("⚔️ Enviar Apuesta", type="primary", use_container_width=True):
            
            # Ejecutar inferencia con la ResNet18
            pred_idx, probabilidades, clases = utils.predecir_imagen(modelo, st.session_state.pkmn_imagen)
            
            # Guardar datos en el estado
            st.session_state.pred_idx = pred_idx
            st.session_state.probabilidades = probabilidades
            st.session_state.turno_evaluado = True
            
            clase_ia = clases[pred_idx]
            tipo_correcto = st.session_state.pkmn_tipo_real
            
            st.write("---")
            st.markdown(f"## 📢 ¡Resultados para **{st.session_state.pkmn_nombre}**!")
            
            col_u, col_ia, col_real = st.columns(3)
            with col_u:
                st.info(f"**Tu respuesta:**\n\n{opcion_usuario}")
            with col_ia:
                st.warning(f"**Inferencia IA:**\n\n{clase_ia}")
            with col_real:
                st.success(f"**Tipo Real (API):**\n\n{tipo_correcto}")
            
            # Repartir los puntos basándose estrictamente en la verdad de la API
            puntos_ganados_humano = 0
            puntos_ganados_ia = 0
            
            if opcion_usuario == tipo_correcto:
                st.session_state.puntaje_humano += 1
                puntos_ganados_humano = 1
            if clase_ia == tipo_correcto:
                st.session_state.puntaje_ia += 1
                puntos_ganados_ia = 1
                
            # Mensajes de feedback divertidos
            if puntos_ganados_humano and puntos_ganados_ia:
                st.balloons()
                st.success("✨ ¡Excelente! Ambos sumaron un punto.")
            elif puntos_ganados_humano and not puntos_ganados_ia:
                st.success("🔥 ¡Punto para vos! Le ganaste a la máquina.")
            elif not puntos_ganados_humano and puntos_ganados_ia:
                st.error("🤖 ¡Punto para la IA! El modelo analizó mejor los patrones visuales.")
            else:
                st.info("❌ ¡Ninguno acertó! El Profesor Oak está decepcionado.")
                
            st.caption("Hacé clic arriba en 'Siguiente Pokémon' para la próxima ronda.")
    else:
        st.warning(f"Ya jugaste este turno. El Pokémon era **{st.session_state.pkmn_nombre}** (Tipo: {st.session_state.pkmn_tipo_real}). ¡Pasá de ronda arriba!")

# ==============================================================================
# TAB 2: SECCIÓN DE ANÁLISIS DE PROBABILIDADES
# ==============================================================================
with tab_analisis:
    st.header("🔬 Diagnóstico de Confianza de la ResNet18")
    
    if st.session_state.turno_evaluado and st.session_state.probabilidades is not None:
        clases = utils.CLASSES
        probabilidades = st.session_state.probabilidades
        pred_idx = st.session_state.pred_idx
        
        st.write(f"Análisis del turno actual frente a **{st.session_state.pkmn_nombre}**:")
        st.subheader(f"Decisión final de la IA: Tipo **{clases[pred_idx]}**")
        st.write(f"Confianza de la neurona ganadora: **{(probabilidades[pred_idx].item()*100):.2f}%**")
        
        st.write("---")
        st.markdown("### 📊 Salida de la capa Softmax:")
        
        # Mapear el tensor a un diccionario para graficar
        diccionario_probabilidades = {clases[i]: float(probabilidades[i]) for i in range(len(clases))}
        st.bar_chart(diccionario_probabilidades)
    else:
        st.info("💡 Completá un turno en la pestaña del juego para analizar la respuesta de la red neuronal.")