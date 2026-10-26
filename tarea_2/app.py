"""Dashboard Tarea 2 – Voces y documentos peruanos (solo lee resultados precalculados)."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
D = BASE / "datos"
st.set_page_config(page_title="Voces y documentos peruanos", page_icon="🎙️", layout="wide")


@st.cache_data
def cargar(nombre):
    p = D / nombre
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


docs, trans = cargar("documentos.csv"), cargar("transcripciones.csv")
ext, metr = cargar("extraccion_llm.csv"), cargar("metricas.csv")
mdoc, maud, audit = cargar("metricas_documentos.csv"), cargar("metricas_audio.csv"), cargar("auditoria_llm.csv")
ejemplos = D / "ejemplos"

st.title("¿Pueden convertirse documentos escaneados y entrevistas en audio peruanos en un archivo de texto estructurado y buscable?")
st.markdown(
    "Pipeline: **PaddleOCR / EasyOCR** sobre 25 documentos peruanos (impresos, fotos y manuscritos) · "
    "**faster-whisper / Parakeet** con VAD y diarización sobre audio en español peruano (OpenSLR 73 + Wikimedia Commons) · "
    "seudonimización · extracción de campos JSON con LLM · evaluación CER/WER sobre 10 documentos y 10 clips con referencia manual. "
    "Los modelos se ejecutaron en notebooks; esta app solo explora los resultados.")

c = st.columns(6)
c[0].metric("Documentos procesados", len(docs))
c[1].metric("Minutos de audio", f"{trans.duracion_seg.sum() / 60:.1f}" if len(trans) else "–")


def m(modelo):
    f = metr[metr.modelo.str.contains(modelo, case=False)] if len(metr) else metr
    return f"{f.valor.iloc[0]:.3f}" if len(f) else "–"


c[2].metric("CER PaddleOCR", m("PaddleOCR"))
c[3].metric("CER EasyOCR", m("EasyOCR"))
c[4].metric("WER whisper-small / Parakeet", f"{m('whisper')} / {m('Parakeet')}")
c[5].metric("Acuerdo LLM vs. humano", f"{int(audit.correcto.sum())}/{len(audit)} ({audit.correcto.mean():.0%})" if len(audit) else "pendiente")

t1, t2, t3, t4 = st.tabs(["📊 Resultados", "🔎 Búsqueda", "📄 Documentos", "🎧 Audios"])

with t1:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(docs, x="tipo", color="tipo", title="Documentos por tipo"), use_container_width=True)
    if len(trans):
        b.plotly_chart(px.bar(trans.sort_values("duracion_seg"), x="id", y="duracion_seg", color="origen",
                              title="Duración por clip (s)"), use_container_width=True)
    if len(mdoc):
        g = mdoc.melt(id_vars=["id", "tipo"], value_vars=["CER_paddle", "CER_easyocr"], var_name="modelo", value_name="CER")
        a.plotly_chart(px.bar(g.groupby(["tipo", "modelo"], as_index=False).CER.mean(), x="tipo", y="CER",
                              color="modelo", barmode="group", title="CER por tipo de documento"), use_container_width=True)
    if len(maud):
        g = maud.melt(id_vars=["id", "condicion"], value_vars=["WER_whisper", "WER_parakeet"], var_name="modelo", value_name="WER")
        b.plotly_chart(px.bar(g, x="id", y="WER", color="modelo", barmode="group", hover_data=["condicion"],
                              title="WER por clip de evaluación"), use_container_width=True)
    if len(metr):
        st.subheader("Tabla comparativa")
        st.dataframe(metr, use_container_width=True, hide_index=True)

with t2:
    q = st.text_input("Buscar palabra en textos OCR y ASR", placeholder="Lima, Huáscar, fútbol…")
    col1, col2 = st.columns(2)
    tipos = col1.multiselect("Tipo de documento", sorted(docs.tipo.unique()), default=sorted(docs.tipo.unique()))
    fuentes_a = col2.multiselect("Origen del audio", sorted(trans.origen.unique()), default=sorted(trans.origen.unique()))
    dmax = float(trans.duracion_seg.max())
    rango = st.slider("Duración del audio (s)", 0.0, dmax, (0.0, dmax))
    fd = docs[docs.tipo.isin(tipos)]
    fa = trans[trans.origen.isin(fuentes_a) & trans.duracion_seg.between(*rango)]
    if q:
        fd = fd[fd.texto_ocr.fillna("").str.contains(q, case=False, regex=False)]
        fa = fa[fa.texto_asr.fillna("").str.contains(q, case=False, regex=False)]
    st.plotly_chart(px.bar(pd.DataFrame({"modalidad": ["documentos", "audios"], "coincidencias": [len(fd), len(fa)]}),
                           x="modalidad", y="coincidencias", title=f"Registros que cumplen los filtros{' para «' + q + '»' if q else ''}"),
                    use_container_width=True)
    st.markdown(f"**Documentos ({len(fd)})**")
    st.dataframe(fd[["id", "tipo", "titulo", "modelo_ocr", "texto_ocr"]], use_container_width=True, hide_index=True)
    st.download_button("Descargar documentos filtrados (CSV)", fd.to_csv(index=False).encode("utf-8"), "documentos_filtrados.csv")
    st.markdown(f"**Audios ({len(fa)})**")
    st.dataframe(fa[["id", "origen", "duracion_seg", "n_hablantes", "modelo_asr", "texto_asr"]], use_container_width=True, hide_index=True)
    st.download_button("Descargar transcripciones filtradas (CSV)", fa.to_csv(index=False).encode("utf-8"), "transcripciones_filtradas.csv")


def campos_llm(i):
    f = ext[ext.id == i] if len(ext) else ext
    return json.loads(f.campos_json.iloc[0]) if len(f) else None


with t3:
    sel = st.selectbox("Documento", docs.id + " · " + docs.tipo + " · " + docs.titulo.fillna(""))
    r = docs[docs.id == sel.split(" · ")[0]].iloc[0]
    a, b = st.columns([1, 1])
    img = ejemplos / Path(r.archivo).name
    if img.exists():
        a.image(str(img), caption=f"{r.fuente} — {r.fuente_url}")
    else:
        a.info("Vista previa no publicada para este documento.")
    a.caption(f"Fuente: {r.fuente} · {r.fuente_url}")
    b.markdown(f"**Modelo OCR:** {r.modelo_ocr}")
    b.text_area("Texto OCR", r.texto_ocr, height=260)
    if isinstance(r.get("texto_referencia"), str) and r.texto_referencia:
        b.text_area("Texto de referencia (manual)", r.texto_referencia, height=160)
        if len(mdoc) and r.id in set(mdoc.id):
            mm = mdoc[mdoc.id == r.id].iloc[0]
            b.metric("CER PaddleOCR / EasyOCR", f"{mm.CER_paddle:.3f} / {mm.CER_easyocr:.3f}")
    cj = campos_llm(r.id)
    b.markdown("**Campos estructurados (LLM)**")
    b.json(cj) if cj else b.caption("Sin extracción LLM disponible.")

with t4:
    sel = st.selectbox("Clip", trans.id + " · " + trans.origen + " · " + trans.duracion_seg.astype(str) + " s")
    r = trans[trans.id == sel.split(" · ")[0]].iloc[0]
    a, b = st.columns([1, 1])
    wav = ejemplos / f"{r.id}.mp3"
    if wav.exists():
        a.audio(str(wav))
    else:
        a.info("Audio no incluido en el repositorio; consultar fuente.")
    a.caption(f"{r.titulo} · {r.fuente} · {r.fuente_url}")
    a.metric("Hablantes (diarización)", int(r.n_hablantes))
    a.markdown(f"**Condición:** {r.condicion}")
    b.markdown(f"**Modelo ASR:** {r.modelo_asr}")
    b.text_area("Transcripción ASR (con marcas de hablante si aplica)", r.texto_asr, height=240)
    alt = cargar("transcripciones_parakeet.csv")
    if len(alt) and r.id in set(alt.id):
        b.text_area("Transcripción Parakeet-TDT 0.6B v3", alt[alt.id == r.id].texto.iloc[0], height=160)
    if isinstance(r.get("texto_referencia"), str) and r.texto_referencia:
        b.text_area("Referencia manual", r.texto_referencia, height=140)
        if len(maud) and r.id in set(maud.id):
            mm = maud[maud.id == r.id].iloc[0]
            b.metric("WER whisper / Parakeet", f"{mm.WER_whisper:.3f} / {mm.WER_parakeet:.3f}")
    cj = campos_llm(r.id)
    b.markdown("**Campos estructurados (LLM)**")
    b.json(cj) if cj else b.caption("Sin extracción LLM disponible.")
