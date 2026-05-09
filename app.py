import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------
st.set_page_config(
    page_title="Proyecto de Sustentabilidad",
    page_icon="♻️",
    layout="wide"
)

plt.style.use("seaborn-v0_8-whitegrid")

ARCHIVOS_DEMO = [
    "Proyecto_Final-estadistica.xlsx",
    "data/Proyecto_Final-estadistica.xlsx"
]

# ---------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------
def leer_archivo_demo():
    """Busca un archivo demo dentro del repo."""
    for ruta in ARCHIVOS_DEMO:
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return f.read(), ruta
    return None, None


def limpiar_nombre_columna(nombre, indice):
    """Limpia nombres de columnas vacíos o tipo Unnamed."""
    nombre = str(nombre).strip()
    if nombre == "" or nombre.lower() == "nan" or nombre.lower().startswith("unnamed"):
        return f"columna_{indice + 1}"
    return nombre


def convertir_columnas_numericas_posibles(df):
    """
    Convierte columnas de texto a numéricas si la mayoría de sus valores
    parecen números.
    """
    nuevo = df.copy()

    for col in nuevo.columns:
        if pd.api.types.is_numeric_dtype(nuevo[col]):
            continue

        serie = nuevo[col].astype(str).str.strip()
        serie = serie.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        serie = serie.str.replace(",", "", regex=False)

        convertida = pd.to_numeric(serie, errors="coerce")
        umbral = max(3, int(len(nuevo) * 0.5))

        if convertida.notna().sum() >= umbral:
            nuevo[col] = convertida

    return nuevo


def limpiar_dataframe(df):
    """Elimina filas/columnas vacías y normaliza encabezados."""
    df = df.copy()

    df.columns = [limpiar_nombre_columna(c, i) for i, c in enumerate(df.columns)]
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                "": np.nan,
                "nan": np.nan,
                "None": np.nan
            })

    df = convertir_columnas_numericas_posibles(df)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cargar_excel_desde_bytes(file_bytes):
    """Carga todas las hojas del Excel en un diccionario."""
    libro = pd.ExcelFile(io.BytesIO(file_bytes))
    hojas = {}

    for hoja in libro.sheet_names:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=hoja)
        hojas[hoja] = limpiar_dataframe(df)

    return hojas


def obtener_columnas_numericas(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def obtener_columnas_texto(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def preparar_tabla_frecuencia(df):
    """
    Intenta detectar una tabla tipo:
    categoría + frecuencia
    y devuelve una tabla lista para graficar.
    """
    trabajo = df.copy()

    columnas_utiles = [c for c in trabajo.columns if trabajo[c].notna().sum() > 0]
    if len(columnas_utiles) < 2:
        return None, None, None

    num_cols = [c for c in columnas_utiles if pd.api.types.is_numeric_dtype(trabajo[c])]
    txt_cols = [c for c in columnas_utiles if c not in num_cols]

    x_col = txt_cols[0] if txt_cols else columnas_utiles[0]

    y_col = None
    candidatos_y = [c for c in num_cols if c != x_col]

    if candidatos_y:
        y_col = max(candidatos_y, key=lambda c: trabajo[c].notna().sum())
    else:
        for c in columnas_utiles:
            if c == x_col:
                continue
            convertida = pd.to_numeric(trabajo[c], errors="coerce")
            if convertida.notna().sum() >= 1:
                trabajo[c] = convertida
                y_col = c
                break

    if y_col is None:
        return None, None, None

    tabla = trabajo[[x_col, y_col]].copy()
    tabla[x_col] = tabla[x_col].astype(str).str.strip()
    tabla[y_col] = pd.to_numeric(tabla[y_col], errors="coerce")
    tabla = tabla.dropna()
    tabla = tabla[tabla[x_col] != ""]

    if tabla.empty:
        return None, None, None

    tabla = tabla.groupby(x_col, as_index=False)[y_col].sum()
    tabla = tabla.sort_values(by=y_col, ascending=False).reset_index(drop=True)

    total = tabla[y_col].sum()
    if total > 0:
        tabla["porcentaje"] = (tabla[y_col] / total * 100).round(2)
    else:
        tabla["porcentaje"] = 0.0

    tabla["acumulado"] = tabla[y_col].cumsum()
    return tabla, x_col, y_col


def resumen_hoja(df):
    """Genera un resumen rápido de una hoja."""
    filas, columnas = df.shape
    num_cols = obtener_columnas_numericas(df)
    txt_cols = obtener_columnas_texto(df)

    return {
        "Filas": filas,
        "Columnas": columnas,
        "Columnas numéricas": len(num_cols),
        "Columnas de texto": len(txt_cols),
        "Valores faltantes": int(df.isna().sum().sum())
    }


def describir_numericas(df):
    """Devuelve estadísticas descriptivas de columnas numéricas."""
    num_cols = obtener_columnas_numericas(df)
    if not num_cols:
        return None
    return df[num_cols].describe().T.round(2)


def detectar_graficas_disponibles(df):
    """Detecta qué tipos de gráficas se pueden generar."""
    disponibles = []

    tabla_freq, _, _ = preparar_tabla_frecuencia(df)
    num_cols = obtener_columnas_numericas(df)

    if tabla_freq is not None and len(tabla_freq) >= 2:
        disponibles += [
            "Barras",
            "Barras horizontales",
            "Pastel",
            "Dona",
            "Línea",
            "Área",
            "Ojiva"
        ]

    if len(num_cols) >= 1:
        disponibles += [
            "Histograma",
            "Boxplot",
            "Violin"
        ]

    if len(num_cols) >= 2:
        disponibles += [
            "Dispersión",
            "Heatmap de correlación"
        ]

    return disponibles


def generar_conclusiones_automaticas(df, nombre_hoja):
    """Genera observaciones rápidas para la hoja."""
    conclusiones = []
    tabla_freq, x_col, y_col = preparar_tabla_frecuencia(df)
    num_cols = obtener_columnas_numericas(df)

    if tabla_freq is not None and not tabla_freq.empty:
        top = tabla_freq.iloc[0]
        total = tabla_freq[y_col].sum()
        conclusiones.append(
            f"En la hoja **{nombre_hoja}**, la categoría con mayor frecuencia es "
            f"**{top[x_col]}** con **{top[y_col]}** registros, equivalente a "
            f"**{top['porcentaje']}%** del total."
        )

    if num_cols:
        principal = num_cols[0]
        serie = pd.to_numeric(df[principal], errors="coerce").dropna()
        if not serie.empty:
            conclusiones.append(
                f"La columna numérica principal **{principal}** tiene una media de "
                f"**{serie.mean():.2f}**, un mínimo de **{serie.min():.2f}** y un máximo de "
                f"**{serie.max():.2f}**."
            )

    if not conclusiones:
        conclusiones.append(
            f"La hoja **{nombre_hoja}** contiene datos útiles, aunque no se detectó una estructura "
            f"perfecta para generar todas las conclusiones automáticas."
        )

    return conclusiones


# ---------------------------------------------------
# FUNCIONES DE GRÁFICAS
# ---------------------------------------------------
def grafica_barras(tabla, x_col, y_col, titulo):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tabla[x_col].astype(str), tabla[y_col], color="#2E8B57")
    ax.set_title(titulo)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def grafica_barras_h(tabla, x_col, y_col, titulo):
    tabla2 = tabla.sort_values(y_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(tabla2[x_col].astype(str), tabla2[y_col], color="#4682B4")
    ax.set_title(titulo)
    ax.set_xlabel(y_col)
    ax.set_ylabel(x_col)
    plt.tight_layout()
    return fig


def grafica_pastel(tabla, x_col, y_col, titulo):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        tabla[y_col],
        labels=tabla[x_col].astype(str),
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title(titulo)
    plt.tight_layout()
    return fig


def grafica_dona(tabla, x_col, y_col, titulo):
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        tabla[y_col],
        labels=tabla[x_col].astype(str),
        autopct="%1.1f%%",
        startangle=90
    )
    centro = plt.Circle((0, 0), 0.60, fc="white")
    fig.gca().add_artist(centro)
    ax.set_title(titulo)
    plt.tight_layout()
    return fig


def grafica_linea(tabla, x_col, y_col, titulo):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tabla[x_col].astype(str), tabla[y_col], marker="o", color="#C0392B")
    ax.set_title(titulo)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def grafica_area(tabla, x_col, y_col, titulo):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tabla))
    ax.fill_between(x, tabla[y_col], color="#8E44AD", alpha=0.45)
    ax.plot(x, tabla[y_col], color="#6C3483")
    ax.set_xticks(x)
    ax.set_xticklabels(tabla[x_col].astype(str), rotation=45, ha="right")
    ax.set_title(titulo)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig


def grafica_ojiva(tabla, x_col, y_col, titulo):
    tabla2 = tabla.copy().sort_values(y_col, ascending=False).reset_index(drop=True)
    tabla2["acumulado"] = tabla2[y_col].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tabla2[x_col].astype(str), tabla2["acumulado"], marker="o", color="#D35400")
    ax.set_title(titulo)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Frecuencia acumulada")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def grafica_histograma(df, col, titulo):
    serie = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(serie, bins=10, color="#16A085", edgecolor="black")
    ax.set_title(titulo)
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    return fig


def grafica_boxplot(df, col, titulo):
    serie = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(serie, vert=False)
    ax.set_title(titulo)
    ax.set_xlabel(col)
    plt.tight_layout()
    return fig


def grafica_violin(df, col, titulo):
    serie = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot(serie, showmeans=True, showmedians=True)
    ax.set_title(titulo)
    ax.set_xticks([1])
    ax.set_xticklabels([col])
    plt.tight_layout()
    return fig


def grafica_dispersion(df, x_col, y_col, titulo):
    datos = df[[x_col, y_col]].copy()
    datos[x_col] = pd.to_numeric(datos[x_col], errors="coerce")
    datos[y_col] = pd.to_numeric(datos[y_col], errors="coerce")
    datos = datos.dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(datos[x_col], datos[y_col], alpha=0.7, color="#1F618D")
    ax.set_title(titulo)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig


def grafica_heatmap(df, titulo):
    num_cols = obtener_columnas_numericas(df)
    corr = df[num_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="YlGnBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title(titulo)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def mostrar_figura(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def renderizar_graficas(df, nombre_hoja, graficas_seleccionadas, col_num_1=None, col_num_2=None):
    tabla_freq, x_col, y_col = preparar_tabla_frecuencia(df)

    for graf in graficas_seleccionadas:
        st.markdown(f"### {graf}")

        try:
            if graf == "Barras" and tabla_freq is not None:
                fig = grafica_barras(tabla_freq, x_col, y_col, f"{nombre_hoja} - Barras")
                mostrar_figura(fig)

            elif graf == "Barras horizontales" and tabla_freq is not None:
                fig = grafica_barras_h(tabla_freq, x_col, y_col, f"{nombre_hoja} - Barras horizontales")
                mostrar_figura(fig)

            elif graf == "Pastel" and tabla_freq is not None:
                fig = grafica_pastel(tabla_freq, x_col, y_col, f"{nombre_hoja} - Pastel")
                mostrar_figura(fig)

            elif graf == "Dona" and tabla_freq is not None:
                fig = grafica_dona(tabla_freq, x_col, y_col, f"{nombre_hoja} - Dona")
                mostrar_figura(fig)

            elif graf == "Línea" and tabla_freq is not None:
                fig = grafica_linea(tabla_freq, x_col, y_col, f"{nombre_hoja} - Línea")
                mostrar_figura(fig)

            elif graf == "Área" and tabla_freq is not None:
                fig = grafica_area(tabla_freq, x_col, y_col, f"{nombre_hoja} - Área")
                mostrar_figura(fig)

            elif graf == "Ojiva" and tabla_freq is not None:
                fig = grafica_ojiva(tabla_freq, x_col, y_col, f"{nombre_hoja} - Ojiva")
                mostrar_figura(fig)

            elif graf == "Histograma" and col_num_1 is not None:
                fig = grafica_histograma(df, col_num_1, f"{nombre_hoja} - Histograma de {col_num_1}")
                mostrar_figura(fig)

            elif graf == "Boxplot" and col_num_1 is not None:
                fig = grafica_boxplot(df, col_num_1, f"{nombre_hoja} - Boxplot de {col_num_1}")
                mostrar_figura(fig)

            elif graf == "Violin" and col_num_1 is not None:
                fig = grafica_violin(df, col_num_1, f"{nombre_hoja} - Violin de {col_num_1}")
                mostrar_figura(fig)

            elif graf == "Dispersión" and col_num_1 is not None and col_num_2 is not None:
                fig = grafica_dispersion(df, col_num_1, col_num_2, f"{nombre_hoja} - Dispersión")
                mostrar_figura(fig)

            elif graf == "Heatmap de correlación":
                num_cols = obtener_columnas_numericas(df)
                if len(num_cols) >= 2:
                    fig = grafica_heatmap(df, f"{nombre_hoja} - Heatmap de correlación")
                    mostrar_figura(fig)

        except Exception as e:
            st.warning(f"No se pudo generar la gráfica '{graf}' en la hoja '{nombre_hoja}'. Error: {e}")


# ---------------------------------------------------
# INTERFAZ
# ---------------------------------------------------
st.title("♻️ Proyecto interactivo de estadística y sustentabilidad")
st.write(
    "Esta app te permite cargar tu archivo de Excel, explorar cada hoja, "
    "ver estadísticas, generar gráficas de distintos tipos y revisar todo el proyecto."
)

with st.sidebar:
    st.header("Opciones de carga")
    archivo_subido = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    usar_demo = st.checkbox("Usar archivo demo del repositorio", value=False)

    st.divider()
    modo = st.radio(
        "¿Qué quieres ver?",
        ["Resumen general", "Explorar una hoja", "Proyecto completo"]
    )

    filas_preview = st.slider("Filas a mostrar en vista previa", 5, 30, 10)


file_bytes = None
origen_archivo = None

if archivo_subido is not None:
    file_bytes = archivo_subido.getvalue()
    origen_archivo = "archivo subido por el usuario"
elif usar_demo:
    file_bytes, ruta_demo = leer_archivo_demo()
    if file_bytes is not None:
        origen_archivo = f"archivo demo encontrado en: {ruta_demo}"

if file_bytes is None:
    st.info(
        "Sube un archivo Excel o activa la opción de archivo demo. "
        "Si quieres que la app funcione sin subir nada, guarda tu Excel en el repo "
        "como `Proyecto_Final-estadistica.xlsx` o en `data/Proyecto_Final-estadistica.xlsx`."
    )
    st.stop()

try:
    hojas = cargar_excel_desde_bytes(file_bytes)
except Exception as e:
    st.error(f"No se pudo leer el archivo Excel. Error: {e}")
    st.stop()

nombres_hojas = list(hojas.keys())
total_filas = sum(df.shape[0] for df in hojas.values())
total_columnas = sum(df.shape[1] for df in hojas.values())

st.success(f"Archivo cargado correctamente desde: **{origen_archivo}**")

# ---------------------------------------------------
# MODO: RESUMEN GENERAL
# ---------------------------------------------------
if modo == "Resumen general":
    col1, col2, col3 = st.columns(3)
    col1.metric("Número de hojas", len(nombres_hojas))
    col2.metric("Total de filas", total_filas)
    col3.metric("Total de columnas", total_columnas)

    st.subheader("Hojas detectadas")
    resumen_libro = []

    for nombre, df in hojas.items():
        r = resumen_hoja(df)
        r["Hoja"] = nombre
        resumen_libro.append(r)

    resumen_df = pd.DataFrame(resumen_libro)
    resumen_df = resumen_df[["Hoja", "Filas", "Columnas", "Columnas numéricas", "Columnas de texto", "Valores faltantes"]]
    st.dataframe(resumen_df, use_container_width=True)

    st.subheader("Vista previa rápida por hoja")
    hoja_resumen = st.selectbox("Selecciona una hoja para previsualizar", nombres_hojas)
    st.dataframe(hojas[hoja_resumen].head(filas_preview), use_container_width=True)

    desc = describir_numericas(hojas[hoja_resumen])
    if desc is not None:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("Esta hoja no tiene columnas numéricas claras para describir.")

# ---------------------------------------------------
# MODO: EXPLORAR UNA HOJA
# ---------------------------------------------------
elif modo == "Explorar una hoja":
    hoja = st.selectbox("Selecciona la hoja", nombres_hojas)
    df = hojas[hoja]

    st.subheader(f"Hoja seleccionada: {hoja}")
    st.dataframe(df.head(filas_preview), use_container_width=True)

    csv_descarga = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar esta hoja como CSV",
        data=csv_descarga,
        file_name=f"{hoja}.csv",
        mime="text/csv"
    )

    st.subheader("Resumen de la hoja")
    resumen = resumen_hoja(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filas", resumen["Filas"])
    c2.metric("Columnas", resumen["Columnas"])
    c3.metric("Numéricas", resumen["Columnas numéricas"])
    c4.metric("Texto", resumen["Columnas de texto"])
    c5.metric("Faltantes", resumen["Valores faltantes"])

    desc = describir_numericas(df)
    if desc is not None:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(desc, use_container_width=True)

    st.subheader("Conclusiones rápidas")
    for texto in generar_conclusiones_automaticas(df, hoja):
        st.write(f"- {texto}")

    disponibles = detectar_graficas_disponibles(df)
    num_cols = obtener_columnas_numericas(df)

    if not disponibles:
        st.warning("No se detectaron suficientes datos para generar gráficas automáticas en esta hoja.")
    else:
        st.subheader("Generación interactiva de gráficas")

        col_num_1 = None
        col_num_2 = None

        if len(num_cols) >= 1:
            col_num_1 = st.selectbox("Columna numérica principal", num_cols, key=f"num1_{hoja}")

        if len(num_cols) >= 2:
            opciones_segunda = [c for c in num_cols if c != col_num_1]
            if opciones_segunda:
                col_num_2 = st.selectbox("Segunda columna numérica", opciones_segunda, key=f"num2_{hoja}")

        graficas_elegidas = st.multiselect(
            "Selecciona qué gráficas quieres ver",
            disponibles,
            default=disponibles
        )

        if graficas_elegidas:
            renderizar_graficas(df, hoja, graficas_elegidas, col_num_1, col_num_2)

# ---------------------------------------------------
# MODO: PROYECTO COMPLETO
# ---------------------------------------------------
elif modo == "Proyecto completo":
    st.subheader("Vista integral del proyecto")

    resumen_general = []
    for nombre, df in hojas.items():
        r = resumen_hoja(df)
        r["Hoja"] = nombre
        resumen_general.append(r)

    st.dataframe(pd.DataFrame(resumen_general), use_container_width=True)

    st.info(
        "Debajo verás cada hoja en un bloque desplegable con vista previa, resumen, "
        "conclusiones automáticas y todas las gráficas disponibles."
    )

    for nombre, df in hojas.items():
        with st.expander(f"📄 {nombre}", expanded=False):
            st.write("#### Vista previa")
            st.dataframe(df.head(filas_preview), use_container_width=True)

            st.write("#### Resumen")
            r = resumen_hoja(df)
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Filas", r["Filas"])
            a2.metric("Columnas", r["Columnas"])
            a3.metric("Numéricas", r["Columnas numéricas"])
            a4.metric("Texto", r["Columnas de texto"])
            a5.metric("Faltantes", r["Valores faltantes"])

            desc = describir_numericas(df)
            if desc is not None:
                st.write("#### Estadísticas descriptivas")
                st.dataframe(desc, use_container_width=True)

            st.write("#### Conclusiones automáticas")
            for texto in generar_conclusiones_automaticas(df, nombre):
                st.write(f"- {texto}")

            disponibles = detectar_graficas_disponibles(df)
            num_cols = obtener_columnas_numericas(df)

            col_num_1 = num_cols[0] if len(num_cols) >= 1 else None
            col_num_2 = num_cols[1] if len(num_cols) >= 2 else None

            if disponibles:
                st.write("#### Gráficas")
                renderizar_graficas(df, nombre, disponibles, col_num_1, col_num_2)
            else:
                st.warning("No se pudieron generar gráficas automáticas para esta hoja.")

st.divider()
st.caption(
    "Hecho con Streamlit + Pandas + Matplotlib. "
    "Puedes subir otro archivo Excel o usar uno demo dentro del repositorio."
)
