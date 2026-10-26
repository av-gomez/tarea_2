"""Genera 01_extraccion.ipynb (luego se ejecuta con nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Tarea 2 · Parte 1 — Corpus y extracción (OCR + ASR)

**Pregunta guía:** ¿Pueden convertirse documentos escaneados y entrevistas en audio peruanos en un archivo de texto estructurado y buscable?

| Ítem | Detalle |
|---|---|
| Fecha de extracción | 2026-10-25 |
| Documentos (25) | Wikimedia Commons (PD, CC BY 2.0, CC BY-SA 3.0/4.0) e Internet Archive (impresos peruanos 1791–1901 en dominio público; John Carter Brown Library, UIUC, BHL). Detalle por archivo en `datos/fuentes.csv` |
| Audio | **OpenSLR 73** – *Crowdsourced high-quality Peruvian Spanish speech* (Google, CC BY-SA 4.0) + 6 entrevistas/lecturas en español peruano de **Wikimedia Commons** (CC BY-SA 4.0) |
| OCR | PaddleOCR 3.x, pipeline PP-OCRv5 `lang=es` (versión exacta impresa abajo) · alternativo: EasyOCR (lang=es) |
| ASR | faster-whisper `small` (CPU, int8, VAD Silero, `language=es`) · alternativo: NVIDIA Parakeet-TDT 0.6B v3 (ONNX int8, sherpa-onnx) |
| Diarización | pyannote-segmentation-3.0 + embeddings 3D-Speaker ERes2Net + clustering (sherpa-onnx, sin token de HF) |
| Entorno | Python 3.11 (`uv venv`), equivalente a `uv sync --group docai` / `--group audio` de las sesiones |

El código reutilizable está en `pipeline/` (`ocr.py`, `voz.py`, `preparar_audio.py`). Las inferencias se cachean en `datos/raw/` (no versionado), por lo que el notebook se re-ejecuta rápido.""")

code("""import sys, hashlib, json, re, platform
from pathlib import Path
import pandas as pd, numpy as np
import plotly.express as px, plotly.io as pio
pio.renderers.default = "notebook_connected+png"   # PNG para que los gráficos se vean en GitHub
sys.path.insert(0, "pipeline")
import ocr, voz
import paddleocr, faster_whisper, sherpa_onnx, easyocr
pd.set_option("display.max_colwidth", 120)
print("Python", platform.python_version())
print("PaddleOCR", paddleocr.__version__, "| faster-whisper", faster_whisper.__version__,
      "| sherpa-onnx", sherpa_onnx.__version__, "| EasyOCR", easyocr.__version__)""")

md("## 1. Corpus A — Documentos (25: 8 impresos, 8 fotos, 9 manuscritos)")
code("""cat = pd.read_csv("datos/raw/documentos/catalogo_documentos.csv")
desc_doc = pd.read_csv("datos/raw/documentos/descartes_documentos.csv")
print(cat.tipo.value_counts().to_dict())
print("Candidatos descartados en la recolección:", len(desc_doc))
desc_doc.motivo.str.split(":").str[0].str.strip().value_counts().head(10)""")

md("### Limpieza: duplicados por hash de archivo")
code("""def sha1(p): return hashlib.sha1(Path(p).read_bytes()).hexdigest()
cat["sha1_local"] = cat.archivo.map(sha1)
dups = cat[cat.duplicated("sha1_local", keep="first")]
print("Duplicados exactos:", len(dups))
cat = cat.drop_duplicates("sha1_local")""")

md("""### OCR con PaddleOCR
Parámetros: `lang="es"` (modelo latino PP-OCRv5), sin clasificador de orientación ni *unwarping*, `text_det_limit_side_len=1600`.
Preprocesamiento común (también para EasyOCR): orientación EXIF, RGB, lado mayor ≤ 1600 px.
Se guarda el JSON crudo por documento en `datos/raw/documentos/ocr_crudo/` y recortes de las 3 primeras líneas detectadas en `datos/raw/documentos/recortes/`.""")
code("""res = {r.id: ocr.paddle(r.archivo) for r in cat.itertuples()}
cat["texto_ocr"] = cat.id.map(lambda i: res[i]["texto"])
cat["modelo_ocr"] = cat.id.map(lambda i: res[i]["modelo"])
cat["conf_media"] = cat.id.map(lambda i: np.mean(res[i]["scores"]) if res[i]["scores"] else 0)
cat["n_lineas"] = cat.id.map(lambda i: len(res[i]["lineas"]))
cat["seg_ocr"] = cat.id.map(lambda i: res[i]["segundos_proceso"])
cat.groupby("tipo")[["conf_media", "n_lineas", "seg_ocr"]].mean().round(2)""")

md("""### Descartes por ilegibilidad
Criterio: se marca un documento si el OCR devuelve menos de 20 caracteres **o** confianza media < 0.5. Los marcados se revisaron visualmente: son legibles para una persona (la baja confianza refleja la dificultad del OCR con caligrafía antigua o reflejos), por lo que **se conservan** como casos difíciles y no se descarta ninguno. Además, en la recolección se descartaron 31 candidatos (tabla anterior) y se reemplazaron hasta completar 25.""")
code("""cat["n_car"] = cat.texto_ocr.str.len()
ilegibles = cat[(cat.n_car < 20) | (cat.conf_media < 0.5)]
print("Marcados por baja confianza (revisados y conservados):", len(ilegibles))
ilegibles[["id", "tipo", "n_car", "conf_media"]]""")

md("## 2. Corpus B — Audio")
md("""**OpenSLR 73** trae enunciados cortos (~5 s) de hablantes peruanos. Con `pipeline/preparar_audio.py` se arman **20 clips de ~95 s**, cada uno concatenando enunciados de **un mismo hablante** (10 mujeres, 10 hombres, semilla 42), con 0.6 s de silencio entre enunciados. La transcripción humana oficial de OpenSLR se conserva como `referencia_fuente`.
A esto se suman **6 grabaciones de Wikimedia Commons** (entrevistas callejeras, ferias con ruido/música, una conferencia y una lectura de Wikipedia hablada), que aportan **varios hablantes** y condiciones difíciles.""")
code("""ol = pd.read_csv("datos/raw/audio/catalogo_openslr.csv")
ex = pd.read_csv("datos/raw/audio/extra/catalogo_extra.csv")
ex["archivo_audio"] = "datos/raw/audio/extra/" + ex.id + ".wav"
ex["id_orig"] = ex.id
ex["id"] = [f"clip_{21 + i:03d}" for i in range(len(ex))]
ex["fuente"] = "Wikimedia Commons (" + ex.licencia + ")"
ol["condicion"] = "lectura en estudio (crowdsourcing)"; ol["n_hablantes_esperado"] = 1; ol["titulo"] = "OpenSLR 73 hablante " + ol.hablante_fuente.astype(str)
aud = pd.concat([ol[["id","archivo_audio","fuente","fuente_url","titulo","condicion","n_hablantes_esperado","referencia_fuente"]],
                 ex[["id","archivo_audio","fuente","fuente_url","titulo","condicion","n_hablantes_esperado","id_orig"]]], ignore_index=True)
aud["sha1_local"] = aud.archivo_audio.map(sha1)
print("Duplicados por hash:", aud.duplicated("sha1_local").sum())
aud = aud.drop_duplicates("sha1_local")
aud[["id", "fuente", "condicion", "n_hablantes_esperado"]].tail(8)""")

md("""### ASR: faster-whisper `small` + VAD
`language="es"`, `vad_filter=True` (Silero), `beam_size=5`, `condition_on_previous_text=False` (reduce alucinaciones en bucle). CPU int8.""")
code("""asr = {r.id: voz.whisper(r.archivo_audio) for r in aud.itertuples()}
aud["duracion_seg"] = aud.id.map(lambda i: round(asr[i]["duracion_seg"], 1))
aud["voz_seg"] = aud.id.map(lambda i: round(sum(s["fin"] - s["inicio"] for s in asr[i]["segmentos"]), 1))
aud["rtf_whisper"] = aud.id.map(lambda i: asr[i]["segundos_proceso"] / asr[i]["duracion_seg"])
print(f"Total: {len(aud)} clips, {aud.duracion_seg.sum()/60:.1f} min  |  RTF medio whisper-small CPU: {aud.rtf_whisper.mean():.2f}")""")

md("""### Descartes: audios sin voz
Criterio: se descarta un clip si el VAD detecta < 10 s de voz o el ASR devuelve < 5 palabras.""")
code("""aud["n_palabras_asr"] = aud.id.map(lambda i: len(asr[i]["texto"].split()))
sin_voz = aud[(aud.voz_seg < 10) | (aud.n_palabras_asr < 5)]
print("Audios descartados por falta de voz:", len(sin_voz))
display(sin_voz[["id", "duracion_seg", "voz_seg", "n_palabras_asr"]])
rech = pd.read_csv("datos/raw/audio/extra/rechazados_extra.csv")
print("Candidatos de audio descartados en la recolección:", len(rech))
aud = aud.drop(sin_voz.index)""")

md("""### Diarización (audios multihablante de Commons)
Se diariza la entrevista multihablante `clip_021` (pyannote-seg-3.0 + ERes2Net, número de hablantes fijado a 3 según la descripción del archivo; con umbral automático 0.6 el clustering sobresegmentaba en 19 hablantes, por eso se fija el número). En el resto de grabaciones de Commons `n_hablantes` se toma de los metadatos por falta de tiempo de cómputo y se asigna a cada segmento ASR el hablante con mayor solapamiento temporal. Se exportan SRT y TXT con marcas `[HABLANTE_n]`.""")
code("""aud["n_hablantes"] = aud.n_hablantes_esperado.astype(int)   # OpenSLR = 1; Commons = según metadatos, salvo los diarizados
DIARIZAR = ["clip_021"]  # entrevista en El Agustino, 3 hablantes (entrevistadora + 2 vendedoras)
for r in aud[aud.id.isin(DIARIZAR)].itertuples():
    turnos = voz.diarizar(r.archivo_audio, n_hablantes=int(r.n_hablantes_esperado))
    segs = voz.asignar_hablantes(asr[r.id]["segmentos"], turnos)
    aud.loc[aud.id == r.id, "n_hablantes"] = len({s["hablante"] for s in segs if s["hablante"] != "HABLANTE_?"})
    asr[r.id]["texto"] = voz.texto_con_hablantes(segs)
aud["texto_asr"] = aud.id.map(lambda i: asr[i]["texto"])
aud["modelo_asr"] = aud.id.map(lambda i: asr[i]["modelo"])
aud[aud.id.isin(ex.id)][["id", "condicion", "n_hablantes_esperado", "n_hablantes"]]""")

md("""## 3. Seudonimización
Procedimiento: (0) el diccionario de personas privadas se guarda en `datos/raw/privados.json` (no versionado) para no publicar los nombres reales; (1) reglas automáticas para DNI (8 dígitos), teléfonos, correos; (2) diccionario de **personas privadas** identificadas en la revisión humana de los textos y metadatos (entrevistados/as de Commons). Personajes históricos y autoridades públicas en placas o documentos oficiales (Grau, Prat, San Martín, alcaldes en placas) **no** se sustituyen porque no son datos personales privados. El procedimiento automático se complementó con lectura de todos los textos de audio de Commons.""")
code("""PRIVADOS = json.load(open("datos/raw/privados.json", encoding="utf-8"))  # diccionario nombre real -> seudónimo; NO versionado (datos/raw/)
REGLAS = {r"\\b\\d{8}\\b": "[DNI]", r"\\b9\\d{2}[ -]?\\d{3}[ -]?\\d{3}\\b": "[TELEFONO]", r"[\\w.]+@[\\w.]+\\.\\w+": "[EMAIL]"}
def seudonimizar(t):
    n = 0
    for real, s in PRIVADOS.items():
        t, k = re.subn(r"\\b" + re.escape(real) + r"\\b", s, t, flags=re.I); n += k
    for pat, s in REGLAS.items():
        t, k = re.subn(pat, s, t); n += k
    return t, n
for df, col in [(cat, "texto_ocr"), (aud, "texto_asr"), (aud, "titulo"), (cat, "titulo")]:
    out = df[col].map(seudonimizar)
    df[col] = out.str[0]; df["entidades_reemplazadas"] = df.get("entidades_reemplazadas", 0) + out.str[1]
print("Entidades reemplazadas — documentos:", cat.entidades_reemplazadas.sum(), "| audio:", aud.entidades_reemplazadas.sum())
# exportación SRT/TXT ya seudonimizada
for r in aud.itertuples():
    segs = [dict(x, texto=seudonimizar(x["texto"])[0]) for x in asr[r.id]["segmentos"]]
    voz.exportar(r.id, segs)
display(aud[aud.entidades_reemplazadas > 0][["id", "entidades_reemplazadas"]])
print(open("datos/raw/audio/salidas/clip_021.srt", encoding="utf-8").read()[:700])""")

md("## 4. Análisis exploratorio")
code("""fig = px.histogram(cat, x="tipo", color="tipo", text_auto=True,
    title="El corpus está balanceado: 8 impresos, 8 fotos y 9 manuscritos peruanos")
fig.update_layout(showlegend=False, yaxis_title="documentos"); fig.show()""")
code("""aud["origen"] = np.where(aud.id.isin(ex.id), "Wikimedia Commons (entrevistas)", "OpenSLR 73 (lectura)")
fig = px.bar(aud.sort_values("duracion_seg"), x="id", y="duracion_seg", color="origen",
    title=f"{aud.duracion_seg.sum()/60:.0f} min de audio: OpenSLR aporta clips homogéneos (~95 s); las entrevistas de Commons son más largas y variables")
fig.update_layout(yaxis_title="duración (s)", xaxis_title=""); fig.show()""")
code("""import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(1, 2, subplot_titles=("Caracteres OCR por documento", "Palabras ASR por clip"))
for t in cat.tipo.unique():
    fig.add_trace(go.Histogram(x=cat[cat.tipo == t].n_car, name=t, nbinsx=12), 1, 1)
fig.add_trace(go.Histogram(x=aud.n_palabras_asr, name="audio", nbinsx=12), 1, 2)
fig.update_layout(barmode="stack", title="Los impresos rinden ~3 veces más texto OCR que fotos y manuscritos; las entrevistas de Commons triplican las palabras de los clips de lectura")
fig.show()""")

md("## 5. Validación: extremos de longitud")
code("""print("OCR más largos:"); display(cat.nlargest(3, "n_car")[["id", "tipo", "n_car", "titulo"]])
print("OCR más cortos:"); display(cat.nsmallest(3, "n_car")[["id", "tipo", "n_car", "titulo"]])
print("Clips más largos:"); display(aud.nlargest(3, "duracion_seg")[["id", "duracion_seg", "n_palabras_asr", "titulo"]])
print("Clips más cortos:"); display(aud.nsmallest(3, "duracion_seg")[["id", "duracion_seg", "n_palabras_asr", "titulo"]])""")
md("Los extremos coinciden con los histogramas: los impresos densos (gacetas, bandos) son los más largos, y las placas breves y manuscritos difíciles los más cortos. En audio, las entrevistas de Commons concentran las duraciones extremas.")

md("## 6. Exportación")
code("""cat["texto_referencia"] = ""
aud["idioma"] = "es (español peruano según la fuente)"
aud["texto_referencia"] = ""
documentos = cat[["id", "archivo", "tipo", "fuente", "texto_ocr", "modelo_ocr", "texto_referencia", "titulo", "fuente_url", "conf_media"]]
transcripciones = aud[["id", "archivo_audio", "duracion_seg", "idioma", "texto_asr", "modelo_asr", "n_hablantes", "texto_referencia",
                       "fuente", "fuente_url", "titulo", "condicion", "origen"]]
documentos.to_csv("datos/documentos.csv", index=False, encoding="utf-8")
transcripciones.to_csv("datos/transcripciones.csv", index=False, encoding="utf-8")
# referencia humana de OpenSLR (se usa en la Parte 2)
aud[["id", "referencia_fuente"]].dropna().to_csv("datos/referencia_openslr.csv", index=False, encoding="utf-8")
fuentes = pd.concat([
    pd.DataFrame({"id": cat.id, "modalidad": "documento", "fuente_url": cat.fuente_url, "licencia": cat.licencia,
                  "acceso": "Descarga directa desde la URL (imagen ≤2000 px); atribución: " + cat.autor.astype(str)}),
    pd.DataFrame({"id": aud.id, "modalidad": "audio", "fuente_url": aud.fuente_url,
                  "licencia": np.where(aud.origen.str.startswith("OpenSLR"), "CC BY-SA 4.0", aud.fuente.str.extract(r"\\((.*)\\)")[0]),
                  "acceso": np.where(aud.origen.str.startswith("OpenSLR"),
                                     "Descargar es_pe_female.zip / es_pe_male.zip de openslr.org/73 y ejecutar pipeline/preparar_audio.py",
                                     "Descargar de Wikimedia Commons y convertir a WAV 16 kHz mono con ffmpeg")})])
fuentes.to_csv("datos/fuentes.csv", index=False, encoding="utf-8")
for n, d in [("documentos", documentos), ("transcripciones", transcripciones), ("fuentes", fuentes)]:
    print(n, d.shape)
display(documentos.head()); display(transcripciones.head())""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).resolve().parents[1] / "01_extraccion.ipynb")
