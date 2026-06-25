import streamlit as st

import utils


st.set_page_config(page_title="Pokemon AI Game", page_icon="🎮", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #f4f7fb;
            --panel: rgba(255, 255, 255, 0.84);
            --panel-strong: #ffffff;
            --text: #17212b;
            --muted: #5f6c7b;
            --accent: #2b6cb0;
            --accent-2: #16a34a;
            --accent-3: #ef4444;
            --accent-4: #f59e0b;
            --border: rgba(23, 33, 43, 0.10);
            --shadow: 0 18px 50px rgba(19, 31, 48, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(43, 108, 176, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(22, 163, 74, 0.12), transparent 28%),
                linear-gradient(180deg, #eef3f9 0%, var(--bg) 100%);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.02em;
        }

        .hero {
            background: linear-gradient(135deg, rgba(23, 33, 43, 0.96), rgba(32, 60, 101, 0.92));
            color: white;
            padding: 1.3rem 1.5rem;
            border-radius: 24px;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
        }

        .hero p {
            margin: 0.45rem 0 0;
            color: rgba(255, 255, 255, 0.82);
            font-size: 1rem;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            padding: 0.9rem 1rem;
            border-radius: 18px;
        }

        [data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 600;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 800;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1.1rem 1.2rem;
            margin: 1rem 0;
        }

        .section-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .result-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.4rem;
        }

        .pill-ok { background: rgba(22, 163, 74, 0.14); color: #15803d; }
        .pill-bad { background: rgba(239, 68, 68, 0.14); color: #b91c1c; }
        .pill-warn { background: rgba(245, 158, 11, 0.14); color: #b45309; }
        .pill-info { background: rgba(43, 108, 176, 0.14); color: #1d4ed8; }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 10px 26px rgba(19, 31, 48, 0.06);
            overflow: hidden;
        }

        div[data-testid="stExpander"] details {
            padding: 0.15rem 0.2rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        button[kind="primary"] {
            border-radius: 14px !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #2b6cb0, #1d4ed8) !important;
            border: none !important;
        }

        button[kind="secondary"] {
            border-radius: 14px !important;
            font-weight: 700 !important;
        }

        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #2b6cb0, #16a34a);
        }
    </style>
    """,
    unsafe_allow_html=True,
)



RUTA_MEJOR_MODELO = "resnet34_layer4_adamw.pt"


@st.cache_resource
def get_model():
    return utils.cargar_modelo_ganador(RUTA_MEJOR_MODELO)


modelo, mensaje, carga_exitosa = get_model()
if not carga_exitosa:
    st.error(mensaje)
    st.stop()


def inicializar_estado():
    if "num_pokemon" not in st.session_state:
        st.session_state.num_pokemon = 20
    if "puntaje_humano" not in st.session_state:
        st.session_state.puntaje_humano = 0
    if "puntaje_ia" not in st.session_state:
        st.session_state.puntaje_ia = 0
    if "lote_id" not in st.session_state:
        st.session_state.lote_id = 1
    if "lote_pokemon" not in st.session_state:
        st.session_state.lote_pokemon = utils.obtener_lote_pokemon(st.session_state.num_pokemon)
    if "resultados_lote" not in st.session_state:
        st.session_state.resultados_lote = None


def cambiar_cantidad():
    st.session_state.lote_pokemon = utils.obtener_lote_pokemon(st.session_state.num_pokemon)
    st.session_state.resultados_lote = None


def reiniciar_lote():
    st.session_state.lote_id += 1
    st.session_state.lote_pokemon = utils.obtener_lote_pokemon(st.session_state.num_pokemon)
    st.session_state.resultados_lote = None


def diccionario_probabilidades(probabilidades, clases):
    return {clases[i]: float(probabilidades[i]) for i in range(len(clases))}


def orden_interesante(resultado):
    if not resultado["acerto_usuario"] and not resultado["acerto_ia"]:
        categoria = 0
    elif resultado["acerto_usuario"] != resultado["acerto_ia"]:
        categoria = 1
    else:
        categoria = 2

    return (categoria, -resultado["confianza_ia"], resultado["nombre"])


def evaluar_lote(lote, lote_id):
    resultados = []
    aciertos_humano = 0
    aciertos_ia = 0

    for idx, pokemon in enumerate(lote):
        respuesta_usuario = st.session_state[f"guess_{lote_id}_{idx}"]
        pred_idx, probabilidades, clases = utils.predecir_imagen(modelo, pokemon["imagen"])
        clase_ia = clases[pred_idx]
        tipo_correcto = pokemon["tipo_real"]
        acerto_usuario = respuesta_usuario == tipo_correcto
        acerto_ia = clase_ia == tipo_correcto

        if acerto_usuario:
            aciertos_humano += 1
        if acerto_ia:
            aciertos_ia += 1

        resultados.append(
            {
                "nombre": pokemon["nombre"],
                "tipo_real": tipo_correcto,
                "respuesta_usuario": respuesta_usuario,
                "respuesta_ia": clase_ia,
                "pred_idx": pred_idx,
                "probabilidades": probabilidades,
                "acerto_usuario": acerto_usuario,
                "acerto_ia": acerto_ia,
                "confianza_ia": float(probabilidades[pred_idx]),
            }
        )

    st.session_state.puntaje_humano += aciertos_humano
    st.session_state.puntaje_ia += aciertos_ia
    st.session_state.resultados_lote = sorted(resultados, key=orden_interesante)


inicializar_estado()

tab_juego, tab_analizador = st.tabs(["🎮 Desafío Pokémon", "🔍 Analizar tu Imagen"])

with tab_juego:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Pokemon Streamlit Arena</div>
            <h1>Humano vs. IA: Desafio Pokemon</h1>
            <p>Competí contra la ResNet34 con un lote de Pokemon al mismo tiempo y revisá el análisis completo en una sola pantalla.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_score1, col_score2, col_cant, col_accion = st.columns([1, 1, 1.5, 1.5])
    with col_score1:
        st.metric("Tu puntaje", st.session_state.puntaje_humano)
    with col_score2:
        st.metric("Puntaje IA", st.session_state.puntaje_ia)
    with col_cant:
        st.slider(
            "Cantidad de Pokemon",
            min_value=5,
            max_value=20,
            key="num_pokemon",
            on_change=cambiar_cantidad,
        )
    with col_accion:
        st.write("")
        st.button(
            f"Generar {st.session_state.num_pokemon} Pokemon nuevos",
            use_container_width=True,
            on_click=reiniciar_lote,
        )

    st.markdown(
        """
        <div class="panel">
            <div class="section-label">Modo de juego</div>
            <p style="margin:0;color:var(--muted);font-size:1rem;">
                Elegí un tipo para cada uno de los Pokemon y luego evaluá el lote. Los resultados quedan ordenados por sorpresa:
                primero los casos donde nadie acertó o hubo desacuerdo, y después los aciertos limpios.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("lote_form"):
        lote = st.session_state.lote_pokemon
        st.markdown('<div class="section-label">Seleccion del lote</div>', unsafe_allow_html=True)
        st.subheader(f"Armá tus {len(lote)} respuestas")
        st.caption("Cada tarjeta tiene una imagen y tu predicción para ese Pokemon.")

        for fila_inicio in range(0, len(lote), 4):
            columnas = st.columns(4)
            for col_offset, pokemon in enumerate(lote[fila_inicio:fila_inicio + 4]):
                idx = fila_inicio + col_offset
                with columnas[col_offset]:
                    with st.container(border=True):
                        st.markdown(
                            f"<span class='result-pill pill-info'>#{idx + 1}</span>"
                            f"<strong>{pokemon['nombre']}</strong>",
                            unsafe_allow_html=True,
                        )
                        st.image(pokemon["imagen"], use_container_width=True)
                        st.selectbox(
                            "Tu tipo",
                            utils.CLASSES,
                            key=f"guess_{st.session_state.lote_id}_{idx}",
                            disabled=(st.session_state.resultados_lote is not None),
                        )

        enviar = st.form_submit_button(
            "Evaluar lote completo",
            use_container_width=True,
            type="primary",
            disabled=(st.session_state.resultados_lote is not None),
        )

    if enviar:
        with st.spinner(f"Evaluando los {len(st.session_state.lote_pokemon)} Pokemon..."):
            evaluar_lote(st.session_state.lote_pokemon, st.session_state.lote_id)
        st.rerun()

    if st.session_state.resultados_lote:
        resultados = st.session_state.resultados_lote
        aciertos_humano = sum(1 for r in resultados if r["acerto_usuario"])
        aciertos_ia = sum(1 for r in resultados if r["acerto_ia"])
        total = len(resultados)
        precision_humana = (aciertos_humano / total) if total else 0
        precision_ia = (aciertos_ia / total) if total else 0
        ganador = "Empate"
        if aciertos_humano > aciertos_ia:
            ganador = "Vos"
        elif aciertos_ia > aciertos_humano:
            ganador = "IA"

        st.markdown('<div class="section-label">Resultados</div>', unsafe_allow_html=True)
        st.markdown("## Resultados del lote")
        st.caption("Acá tenés el resumen general y, debajo, el detalle Pokemon por Pokemon con su gráfico.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Aciertos tuyos", aciertos_humano, f"{precision_humana * 100:.1f}%")
        with c2:
            st.metric("Aciertos IA", aciertos_ia, f"{precision_ia * 100:.1f}%")
        with c3:
            st.metric("Pokemon analizados", total)
        with c4:
            st.metric("Ganador del lote", ganador)

        progreso_humano, progreso_ia = st.columns(2)
        with progreso_humano:
            st.write("Tu tasa de acierto")
            st.progress(precision_humana)
            st.caption(f"{precision_humana * 100:.1f}%")
        with progreso_ia:
            st.write("Tasa de acierto de la IA")
            st.progress(precision_ia)
            st.caption(f"{precision_ia * 100:.1f}%")

        st.write("### Resumen rapido")
        resumen = [
            {
                "Pokemon": r["nombre"],
                "Tu respuesta": r["respuesta_usuario"],
                "IA": r["respuesta_ia"],
                "Real": r["tipo_real"],
                "Tu resultado": "Correcto" if r["acerto_usuario"] else "Incorrecto",
                "IA correcta": "Si" if r["acerto_ia"] else "No",
            }
            for r in resultados
        ]
        st.dataframe(resumen, use_container_width=True, hide_index=True)

        st.write("### Detalle por Pokemon")
        for idx, resultado in enumerate(resultados, start=1):
            if not resultado["acerto_usuario"] and not resultado["acerto_ia"]:
                badge = '<span class="result-pill pill-bad">Nadie acertó</span>'
            elif resultado["acerto_usuario"] != resultado["acerto_ia"]:
                badge = '<span class="result-pill pill-warn">Hubo desacuerdo</span>'
            else:
                badge = '<span class="result-pill pill-ok">Ambos acertaron</span>'
            resumen_linea = (
                f"Tu respuesta: {resultado['respuesta_usuario']} | "
                f"IA: {resultado['respuesta_ia']} | "
                f"Real: {resultado['tipo_real']}"
            )
            with st.expander(f"{idx}. {resultado['nombre']}  -  {resumen_linea}", expanded=False):
                st.markdown(badge, unsafe_allow_html=True)
                cabecera_1, cabecera_2, cabecera_3, cabecera_4 = st.columns(4)
                with cabecera_1:
                    st.metric("Tu respuesta", resultado["respuesta_usuario"])
                with cabecera_2:
                    st.metric("IA", resultado["respuesta_ia"])
                with cabecera_3:
                    st.metric("Tipo real", resultado["tipo_real"])
                with cabecera_4:
                    estado = "OK" if resultado["acerto_usuario"] else "Fallo"
                    st.metric("Tu acierto", estado)

                col_estado, col_grafico = st.columns([1, 2])
                with col_estado:
                    if resultado["acerto_usuario"] and resultado["acerto_ia"]:
                        st.success("Los dos acertaron.")
                    elif resultado["acerto_usuario"] and not resultado["acerto_ia"]:
                        st.success("Vos acertaste y la IA no.")
                    elif not resultado["acerto_usuario"] and resultado["acerto_ia"]:
                        st.error("La IA acertó y vos no.")
                    else:
                        st.info("Ninguno acertó.")

                    st.caption(f"Confianza IA: {resultado['confianza_ia'] * 100:.2f}%")
                    st.progress(resultado["confianza_ia"])

                with col_grafico:
                    st.bar_chart(diccionario_probabilidades(resultado["probabilidades"], utils.CLASSES))

with tab_analizador:
    st.markdown(
        """
        <div class="hero" style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.96), rgba(15, 23, 42, 0.92));">
            <div class="section-label">Herramienta de Inferencia</div>
            <h1>Analizador de Pokémon</h1>
            <p>Subí una imagen o usá tu cámara para ver qué tipo de Pokémon predice la red neuronal (ResNet34) junto con sus probabilidades.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_uploader, col_previsualizacion = st.columns([1.2, 1])

    with col_uploader:
        with st.container(border=True):
            st.markdown('<div class="section-label">Entrada de imagen</div>', unsafe_allow_html=True)
            metodo = st.radio(
                "Seleccioná el método de entrada:",
                ["Subir archivo", "Usar cámara"],
                horizontal=True,
            )
            
            imagen_subida = None
            if metodo == "Subir archivo":
                st.write("Formatos soportados: PNG, JPG, JPEG")
                imagen_subida = st.file_uploader(
                    "Arrastrá y soltá una imagen aquí o hacé clic para buscar",
                    type=["png", "jpg", "jpeg"],
                    label_visibility="collapsed",
                )
            else:
                imagen_subida = st.camera_input(
                    "Tomar foto con tu cámara",
                    label_visibility="collapsed",
                )

    with col_previsualizacion:
        if imagen_subida is not None:
            from PIL import Image
            imagen_pil = Image.open(imagen_subida).convert("RGB")
            with st.container(border=True):
                st.markdown('<div class="section-label">Previsualización</div>', unsafe_allow_html=True)
                st.image(imagen_pil, caption="Imagen seleccionada", use_container_width=True)
        else:
            st.info("Subí una imagen o usá la cámara para ver la previsualización y el análisis de la IA.")

    if imagen_subida is not None:
        st.markdown("---")
        with st.container(border=True):
            st.markdown('<div class="section-label">Análisis de la IA</div>', unsafe_allow_html=True)
            st.subheader("Resultados de la Inferencia")

            with st.spinner("Infiriendo el tipo de Pokémon..."):
                pred_idx, probabilidades, clases = utils.predecir_imagen(modelo, imagen_pil)
                tipo_predicho = clases[pred_idx]
                confianza = float(probabilidades[pred_idx])

            c_pred1, c_pred2 = st.columns([1, 2])

            with c_pred1:
                st.metric("Tipo predicho", tipo_predicho)
                st.metric("Confianza de la IA", f"{confianza * 100:.2f}%")
                st.progress(confianza)

            with c_pred2:
                st.write("### Distribución de probabilidades")
                st.bar_chart(diccionario_probabilidades(probabilidades, clases))
