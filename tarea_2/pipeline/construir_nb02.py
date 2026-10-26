"""Genera 02_evaluacion.ipynb."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Tarea 2 · Parte 2 — Evaluación, extracción estructurada y comparación de modelos

**Protocolo (fijado antes de calcular métricas):**
- Muestra de **10 documentos** (3 impresos, 4 fotos, 3 manuscritos) y **10 clips** de OpenSLR 73 (10 hablantes distintos, 5 mujeres y 5 hombres), todos del corpus de la Parte 1. Las grabaciones de Commons (ruido, música, varios hablantes) no tienen referencia manual completa, por lo que se analizan aparte como *caso difícil* mediante el desacuerdo entre modelos.
- Las configuraciones de todos los modelos son las de la Parte 1 y no se tocaron después de ver métricas.
- Normalización idéntica para referencia e hipótesis: minúsculas, sin tildes, sin puntuación, espacios colapsados; en audio se eliminan marcas `[HABLANTE_n]`.
- **Origen de las referencias:** en audio, el texto humano de OpenSLR 73 (guion leído por cada hablante, revisado contra el audio); en documentos, transcripción hecha mirando la imagen ampliada, revisada en dos pasadas (columna `estado_revision`). **Importante:** las referencias de documentos fueron transcritas por el asistente de IA, no por una persona; son independientes de los tres OCR evaluados, pero no son una transcripción humana.""")

code("""import sys, json, re, unicodedata
import pandas as pd, numpy as np, jiwer
import plotly.express as px, plotly.io as pio
pio.renderers.default = "notebook_connected+png"
sys.path.insert(0, "pipeline")
import ocr, voz
docs = pd.read_csv("datos/documentos.csv"); trans = pd.read_csv("datos/transcripciones.csv")
def norm(t):
    t = re.sub(r"\\[HABLANTE_[^\\]]*\\]", " ", str(t))
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-zñ0-9 ]", " ", t.replace("ñ", "ñ"))
    return re.sub(r"\\s+", " ", t).strip()
cer = lambda r, h: jiwer.cer(norm(r), norm(h)) if norm(h) else 1.0
wer = lambda r, h: jiwer.wer(norm(r), norm(h)) if norm(h) else 1.0""")

md("## 1. Muestra de evaluación y referencias manuales")
code("""ref_doc = pd.read_csv("datos/referencias_documentos.csv")
EVAL_DOCS = ref_doc.id.tolist()
ref_ol = pd.read_csv("datos/referencia_openslr.csv")
ref_cm = pd.read_csv("datos/referencias_audio_commons.csv") if __import__("os").path.exists("datos/referencias_audio_commons.csv") else pd.DataFrame(columns=["id","texto_referencia","estado_revision"])
EVAL_OL = ["clip_001", "clip_003", "clip_005", "clip_007", "clip_009", "clip_011", "clip_013", "clip_015", "clip_017", "clip_019"]  # 5 mujeres + 5 hombres, hablantes distintos
ref_aud = pd.concat([ref_ol[ref_ol.id.isin(EVAL_OL)].rename(columns={"referencia_fuente": "texto_referencia"}).assign(estado_revision="transcripción humana OpenSLR"),
                     ref_cm], ignore_index=True)
EVAL_AUD = ref_aud.id.tolist()
docs = docs.drop(columns="texto_referencia").merge(ref_doc[["id", "texto_referencia"]], on="id", how="left")
trans = trans.drop(columns="texto_referencia").merge(ref_aud[["id", "texto_referencia"]], on="id", how="left")
docs.to_csv("datos/documentos.csv", index=False, encoding="utf-8"); trans.to_csv("datos/transcripciones.csv", index=False, encoding="utf-8")
print("Documentos de evaluación:", docs[docs.id.isin(EVAL_DOCS)].tipo.value_counts().to_dict())
print("Clips de evaluación:", len(EVAL_AUD)); display(ref_aud[["id", "estado_revision"]])""")

md("## 2. Línea base: PaddleOCR (sin personalización) y faster-whisper `small` CPU")
code("""ed = docs[docs.id.isin(EVAL_DOCS)].copy()
ed["CER_paddle"] = [cer(r.texto_referencia, ocr.paddle(r.archivo)["texto"]) for r in ed.itertuples()]
ea = trans[trans.id.isin(EVAL_AUD)].copy()
ea["WER_whisper"] = [wer(r.texto_referencia, voz.whisper(r.archivo_audio)["texto"]) for r in ea.itertuples()]
ea["RTF_whisper"] = [voz.whisper(r.archivo_audio)["segundos_proceso"] / r.duracion_seg for r in ea.itertuples()]
print(f"LÍNEA BASE  CER PaddleOCR = {ed.CER_paddle.mean():.3f}   |   WER whisper-small = {ea.WER_whisper.mean():.3f}")""")

md("## 3. Modelos alternativos sobre las mismas muestras")
md("""- **Documentos:** dos OCR alternativos: **Tesseract 5** (LSTM, `spa` de *tessdata_best*, `--psm 3`) y **EasyOCR** (detector CRAFT + reconocedor CRNN, `lang=es`). Mismo preprocesamiento que PaddleOCR (EXIF, RGB, lado ≤ 1600 px). Los tiempos se midieron con la CPU sin otros procesos.
- **Audio:** NVIDIA Parakeet-TDT 0.6B v3 (multilingüe, incluye español) en ONNX int8 vía sherpa-onnx, con VAD Silero. RTF = tiempo de proceso / duración.""")
code("""ed["CER_easyocr"] = [cer(r.texto_referencia, ocr.easyocr(r.archivo)["texto"]) for r in ed.itertuples()]
ed["CER_tesseract"] = [cer(r.texto_referencia, ocr.tesseract(r.archivo)["texto"]) for r in ed.itertuples()]
ed["seg_tesseract"] = [ocr.tesseract(r.archivo)["segundos_proceso"] for r in ed.itertuples()]
ea["WER_parakeet"] = [wer(r.texto_referencia, voz.parakeet(r.archivo_audio)["texto"]) for r in ea.itertuples()]
ea["RTF_parakeet"] = [voz.parakeet(r.archivo_audio)["segundos_proceso"] / r.duracion_seg for r in ea.itertuples()]
ed["seg_paddle"] = [ocr.paddle(r.archivo)["segundos_proceso"] for r in ed.itertuples()]
ed["seg_easyocr"] = [ocr.easyocr(r.archivo)["segundos_proceso"] for r in ed.itertuples()]
display(ed[["id", "tipo", "CER_paddle", "CER_easyocr", "CER_tesseract", "seg_paddle", "seg_easyocr", "seg_tesseract"]].round(3))
display(ea[["id", "condicion", "WER_whisper", "WER_parakeet", "RTF_whisper", "RTF_parakeet"]].round(3))""")

md("## 4. Tabla comparativa y casos difíciles")
code("""tabla = pd.DataFrame([
    {"modalidad": "documentos", "modelo": "PaddleOCR PP-OCRv5 (base)", "métrica": "CER", "valor": ed.CER_paddle.mean(), "costo": f"{ed.seg_paddle.mean():.1f} s/doc CPU"},
    {"modalidad": "documentos", "modelo": "EasyOCR (alternativo)", "métrica": "CER", "valor": ed.CER_easyocr.mean(), "costo": f"{ed.seg_easyocr.mean():.1f} s/doc CPU"},
    {"modalidad": "documentos", "modelo": "Tesseract 5 spa (alternativo)", "métrica": "CER", "valor": ed.CER_tesseract.mean(), "costo": f"{ed.seg_tesseract.mean():.1f} s/doc CPU"},
    {"modalidad": "audio", "modelo": "faster-whisper small (base)", "métrica": "WER", "valor": ea.WER_whisper.mean(), "costo": f"RTF {ea.RTF_whisper.mean():.2f}"},
    {"modalidad": "audio", "modelo": "Parakeet-TDT 0.6B v3 (alternativo)", "métrica": "WER", "valor": ea.WER_parakeet.mean(), "costo": f"RTF {ea.RTF_parakeet.mean():.2f}"}])
tabla["valor"] = tabla.valor.round(3)
tabla.to_csv("datos/metricas.csv", index=False, encoding="utf-8")
ed[["id","tipo","CER_paddle","CER_easyocr","CER_tesseract"]].to_csv("datos/metricas_documentos.csv", index=False)
ea[["id","condicion","WER_whisper","WER_parakeet","RTF_whisper","RTF_parakeet"]].to_csv("datos/metricas_audio.csv", index=False)
tabla""")
code("""fig = px.bar(ed.melt(id_vars=["id","tipo"], value_vars=["CER_paddle","CER_easyocr","CER_tesseract"], var_name="modelo", value_name="CER")
             .groupby(["tipo","modelo"], as_index=False).CER.mean(), x="tipo", y="CER", color="modelo", barmode="group",
             title="El CER se dispara en manuscritos para ambos OCR: ninguno está entrenado en letra colonial")
fig.show()
ea["grupo"] = np.where(ea.id.isin(["clip_001","clip_003","clip_005","clip_007","clip_009"]), "OpenSLR mujeres", "OpenSLR hombres")
fig = px.bar(ea.melt(id_vars=["id","grupo"], value_vars=["WER_whisper","WER_parakeet"], var_name="modelo", value_name="WER")
             .groupby(["grupo","modelo"], as_index=False).WER.mean(), x="grupo", y="WER", color="modelo", barmode="group",
             title="WER por género del hablante en la muestra de OpenSLR")
fig.show()""")
code("""print("Por tipo de documento:"); display(ed.groupby("tipo")[["CER_paddle","CER_easyocr","CER_tesseract"]].mean().round(3))
print("Por tipo de audio:"); display(ea.groupby("grupo")[["WER_whisper","WER_parakeet"]].mean().round(3))""")

md("""### Caso difícil: entrevistas con ruido, música y varios hablantes (Commons)
Sin referencia manual completa no se reporta WER; como indicador se usa el **desacuerdo entre modelos** (WER de Parakeet tomando a whisper como pseudo-referencia). Un valor alto indica que al menos uno de los dos falla. **No es una métrica de exactitud.**""")
code("""PRIVADOS = json.load(open("datos/raw/privados.json", encoding="utf-8"))  # mismo diccionario no versionado del notebook 01
def seud(t):
    for real, s in PRIVADOS.items():
        t = re.sub(r"\\b" + re.escape(real) + r"\\b", s, t, flags=re.I)
    return t
cm = trans[trans.origen.str.startswith("Wikimedia")].copy()
cm["desacuerdo_modelos"] = [wer(voz.whisper(r.archivo_audio)["texto"], voz.parakeet(r.archivo_audio)["texto"]) for r in cm.itertuples()]
ol = trans[trans.id.isin(EVAL_AUD)].copy()
ol["desacuerdo_modelos"] = [wer(voz.whisper(r.archivo_audio)["texto"], voz.parakeet(r.archivo_audio)["texto"]) for r in ol.itertuples()]
print(f"Desacuerdo medio whisper vs Parakeet — OpenSLR: {ol.desacuerdo_modelos.mean():.2f} | Commons: {cm.desacuerdo_modelos.mean():.2f}")
display(cm[["id", "condicion", "n_hablantes", "desacuerdo_modelos"]].round(2))
for r in cm.iloc[[0, 2]].itertuples():
    print("---", r.id, r.condicion)
    print("whisper :", seud(voz.whisper(r.archivo_audio)["texto"])[:300])
    print("parakeet:", seud(voz.parakeet(r.archivo_audio)["texto"])[:300])
pd.DataFrame([{"id": r.id, "texto": seud(voz.parakeet(r.archivo_audio)["texto"])} for r in trans.itertuples()]).to_csv(
    "datos/transcripciones_parakeet.csv", index=False, encoding="utf-8")""")

md("## 5. Extracción estructurada con LLM")
code("""import os
from dotenv import load_dotenv; load_dotenv("../.env"); load_dotenv(".env")
HAY_LLM = bool(os.environ.get("GEMINI_API_KEY"))
print("API key de Gemini disponible:", HAY_LLM)
# gemini-3.1-flash-lite: se usó un único modelo para las 51 extracciones (la cuota gratuita de gemini-3.5-flash se agotó a mitad del proceso)
if HAY_LLM:
    import llm
    filas = [llm.extraer(r.id, r.texto_ocr, "documento") for r in docs.itertuples()] + \\
            [llm.extraer(r.id, r.texto_asr, "audio") for r in trans.itertuples()]
    ext = pd.DataFrame([{"id": f["id"], "modalidad": f["modalidad"], "campos_json": json.dumps(f["campos"], ensure_ascii=False),
                         "modelo_llm": f["modelo_llm"], "fecha": f["fecha"]} for f in filas])
    ext.to_csv("datos/extraccion_llm.csv", index=False, encoding="utf-8")
    display(ext.head())""")

md("## 6. Auditoría humana de 20 extracciones")
md("""Criterios: una extracción es **completamente correcta** si (a) `tema` y `resumen` son fieles al texto, (b) todo campo no nulo está sustentado literalmente en el texto, (c) no se omiten datos explícitos del texto (`fecha`, `emisor`, `monto`) y (d) no hay campos inventados. Muestra: 10 documentos + 10 audios al azar (`DataFrame.sample(10, random_state=7)` en cada corpus). Para `citas_clave` y `lugares_mencionados` se verificó por código que cada cadena aparezca literalmente en el texto.

**Quién revisó:** la revisión campo por campo la hizo el asistente de IA (Claude Code), comparando cada campo con el texto OCR/ASR. La columna `revisor` de `datos/auditoria_llm.csv` lo indica; la estudiante debe confirmarla o corregirla.""")
code("""muestra = pd.concat([docs.sample(10, random_state=7)[["id"]], trans.sample(10, random_state=7)[["id"]]]).id.tolist()
print(muestra)""")
md("""Resultado:""")
code("""if os.path.exists("datos/auditoria_llm.csv"):
    aud_llm = pd.read_csv("datos/auditoria_llm.csv")
    ok = int(aud_llm.correcto.sum())
    print(f"Acuerdo LLM-Humano: {ok}/20 = {ok/20:.0%}")
    display(aud_llm[aud_llm.correcto == 0][["id", "error"]])
else:
    print("Auditoría pendiente")""")

md("## 7. Decisión y reflexión")
code("""print(tabla.to_string(index=False))""")
md("""### Decisión
| Modalidad | Elección | Por qué |
|---|---|---|
| Documentos | **PaddleOCR PP-OCRv5 mobile** | Menor CER global (0.475 frente a 0.529 de EasyOCR y 0.673 de Tesseract) y 5 veces más rápido que EasyOCR en CPU (5.6 frente a 28.6 s/doc). Tesseract es el más rápido (2.5 s/doc), pero su CER en fotos duplica al de PaddleOCR. |
| Audio | **faster-whisper small** como modelo principal; **Parakeet** como alternativa si importa la velocidad | En lectura limpia los dos quedan casi empatados (WER 0.032 frente a 0.039; la diferencia es de unas pocas palabras sobre ~1 400). Parakeet es ~3 veces más rápido (RTF 0.17 frente a 0.48). Whisper entrega puntuación y segmentos más útiles para el SRT y la diarización. |

### Reflexión por tipo
- **Fotos (CER 0.22 con PaddleOCR, el mejor caso).** Las placas tienen letra grande de molde. El error se concentra en la placa de bronce con reflejos (doc_014, CER 0.36) y en la leyenda latina curva (doc_016, 0.43). La placa de Grau, nítida, baja a 0.04.
- **Impresos (0.48).** El CER es más alto de lo esperado porque los 3 impresos evaluados tienen **partes manuscritas o tipografía decorativa**: los datos escritos a mano en el recibo de 1895, la caligrafía inglesa del diploma de 1921 y los textos superpuestos con sello de la acción de 1878. El OCR lee bien las partes tipográficas y falla en lo demás.
- **Manuscritos (0.74–0.81, el peor caso).** Ningún modelo está entrenado en letra colonial de los siglos XVIII y XIX ni en sus abreviaturas. Por eso los tres OCR producen texto casi inutilizable. EasyOCR queda algo mejor, pero el error sigue siendo de 0.74. Para este tipo haría falta un modelo de HTR especializado (por ejemplo, Transkribus o TrOCR afinado) o una transcripción humana.
- **Audio en estudio frente a audio de calle.** Con lectura limpia los dos ASR cometen menos de 4 % de error. En las entrevistas de Commons el desacuerdo entre modelos sube de 0.04 a 0.36 de media, y llega a 0.73 en la conferencia con audio a −40 dB y a 0.66 en la feria con música. Es la señal de que el ruido, la música y el habla superpuesta degradan mucho la transcripción.
- **LLM.** Acertó en 16 de 20 casos. Los errores aparecen cuando el OCR es ruidoso: copió basura como si fuera una fecha, inventó una fecha, confundió el emisor de un documento citado con el del documento, y una vez parafraseó una cita que debía ser textual. Ante un OCR ilegible (doc_017 y doc_022) respondió bien: dejó los campos en `null` y no inventó nada.""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).resolve().parents[1] / "02_evaluacion.ipynb")
